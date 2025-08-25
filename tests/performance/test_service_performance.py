"""
Performance tests for claude-tui services.

Tests performance characteristics including:
- Throughput under load
- Memory usage patterns
- Concurrent operation scaling
- Response time consistency
- Resource utilization efficiency
"""

import pytest
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import patch

from services.ai_service import AIService
from services.project_service import ProjectService
from services.task_service import TaskService
from services.validation_service import ValidationService
from tests.mocks.mock_dependencies import MockServiceContext
from tests.conftest import PerformanceMonitor


class TestAIServicePerformance:
    """Performance tests for AI Service."""
    
    @pytest.mark.performance
    async def test_code_generation_throughput(self, performance_test_config):
        """Test AI service code generation throughput."""
        with MockServiceContext() as mocks:
            # Configure fast AI mock
            mocks['ai_interface'].execution_time = 0.01
            
            service = AIService()
            await service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Generate code requests
            tasks = []
            for i in range(performance_test_config['test_iterations']):
                task = service.generate_code(
                    f"Generate function {i}",
                    'python',
                    use_cache=False  # Disable caching for throughput test
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            monitor.stop()
            
            # Performance assertions
            assert len(results) == performance_test_config['test_iterations']
            assert all('code' in result for result in results)
            
            # Throughput calculation
            throughput = len(results) / monitor.execution_time
            assert throughput > 50  # Should handle >50 requests/second
            
            print(f"AI Service Throughput: {throughput:.2f} requests/second")
    
    @pytest.mark.performance
    async def test_concurrent_code_generation_scaling(self, performance_test_config):
        """Test AI service scaling with concurrent requests."""
        with MockServiceContext() as mocks:
            service = AIService()
            await service.initialize()
            
            # Test different concurrency levels
            concurrency_levels = [1, 5, 10, 20, 50]
            results = {}
            
            for concurrency in concurrency_levels:
                monitor = PerformanceMonitor()
                monitor.start()
                
                # Create concurrent requests
                tasks = [
                    service.generate_code(f"Function {i}", 'python')
                    for i in range(concurrency)
                ]
                
                await asyncio.gather(*tasks)
                monitor.stop()
                
                results[concurrency] = {
                    'execution_time': monitor.execution_time,
                    'throughput': concurrency / monitor.execution_time
                }
            
            # Verify scaling characteristics
            assert results[1]['throughput'] > 0
            assert results[50]['throughput'] > results[1]['throughput'] * 10  # Should scale well
            
            print("Concurrency Scaling Results:", results)
    
    @pytest.mark.performance
    async def test_ai_service_memory_usage(self, performance_test_config):
        """Test AI service memory usage patterns."""
        with MockServiceContext() as mocks:
            service = AIService()
            await service.initialize()
            
            # Measure initial memory
            gc.collect()  # Force garbage collection
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Perform many operations
            for i in range(100):
                await service.generate_code(f"Function {i}", 'python')
                
                # Force periodic garbage collection
                if i % 10 == 0:
                    gc.collect()
            
            # Measure final memory
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory should not grow excessively
            assert memory_increase < performance_test_config['memory_limit_mb']
            
            print(f"Memory usage - Initial: {initial_memory:.2f}MB, "
                  f"Final: {final_memory:.2f}MB, "
                  f"Increase: {memory_increase:.2f}MB")


class TestProjectServicePerformance:
    """Performance tests for Project Service."""
    
    @pytest.mark.performance
    async def test_project_creation_performance(self, performance_test_config, temp_directory):
        """Test project creation performance."""
        with MockServiceContext() as mocks:
            service = ProjectService()
            await service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create multiple projects
            tasks = []
            for i in range(performance_test_config['test_iterations'] // 10):  # Fewer iterations for file ops
                project_path = temp_directory / f"perf_project_{i}"
                task = service.create_project(
                    name=f"Performance Project {i}",
                    path=project_path,
                    project_type="python",
                    initialize_git=False,
                    create_venv=False
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            monitor.stop()
            
            # Performance assertions
            assert len(results) == performance_test_config['test_iterations'] // 10
            assert all('project_id' in result for result in results)
            
            # Project creation should be reasonably fast
            avg_time_per_project = monitor.execution_time / len(results)
            assert avg_time_per_project < 0.5  # <500ms per project
            
            print(f"Project creation - Average time: {avg_time_per_project:.3f}s per project")
    
    @pytest.mark.performance
    async def test_project_validation_performance(self, performance_test_config, temp_directory):
        """Test project validation performance with various project sizes."""
        with MockServiceContext() as mocks:
            service = ProjectService()
            await service.initialize()
            
            # Create projects with different complexity
            project_sizes = [
                {'files': 10, 'dirs': 3},
                {'files': 100, 'dirs': 10},
                {'files': 1000, 'dirs': 50}
            ]
            
            results = {}
            
            for i, size in enumerate(project_sizes):
                # Create test project structure
                project_path = temp_directory / f"validation_project_{i}"
                project_path.mkdir()
                
                # Create directories
                for d in range(size['dirs']):
                    (project_path / f"dir_{d}").mkdir()
                
                # Create files
                for f in range(size['files']):
                    (project_path / f"file_{f}.py").write_text(f"# File {f}")
                
                # Load project
                project_result = await service.load_project(project_path)
                project_id = project_result['project_id']
                
                # Measure validation time
                monitor = PerformanceMonitor()
                monitor.start()
                
                validation_result = await service.validate_project(project_id)
                
                monitor.stop()
                
                results[f"{size['files']}_files"] = {
                    'validation_time': monitor.execution_time,
                    'files_per_second': size['files'] / monitor.execution_time,
                    'is_valid': validation_result['is_valid']
                }
            
            # Validation time should scale reasonably with project size
            assert all(r['validation_time'] < 2.0 for r in results.values())  # <2s for any size
            assert all(r['files_per_second'] > 100 for r in results.values())  # >100 files/second
            
            print("Project validation performance:", results)


class TestTaskServicePerformance:
    """Performance tests for Task Service."""
    
    @pytest.mark.performance
    async def test_task_execution_throughput(self, performance_test_config):
        """Test task service execution throughput."""
        with MockServiceContext() as mocks:
            # Configure fast task engine
            mocks['task_engine'].set_execution_time(0.01)
            
            service = TaskService()
            await service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create and execute tasks
            for i in range(performance_test_config['test_iterations']):
                task_result = await service.create_task(
                    name=f"Performance Task {i}",
                    description=f"Task {i} for throughput testing",
                    ai_enabled=False  # Use engine for faster execution
                )
                
                await service.execute_task(task_result['task_id'])
            
            monitor.stop()
            
            # Calculate throughput
            throughput = performance_test_config['test_iterations'] / monitor.execution_time
            
            # Should handle reasonable task throughput
            assert throughput > 20  # >20 tasks/second
            
            print(f"Task Service Throughput: {throughput:.2f} tasks/second")
    
    @pytest.mark.performance
    async def test_concurrent_task_execution(self, performance_test_config):
        """Test concurrent task execution performance."""
        with MockServiceContext() as mocks:
            service = TaskService()
            await service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create tasks
            task_results = []
            for i in range(performance_test_config['concurrent_operations']):
                result = await service.create_task(
                    name=f"Concurrent Task {i}",
                    description=f"Task {i} for concurrent testing"
                )
                task_results.append(result)
            
            # Execute all tasks concurrently
            execution_tasks = [
                service.execute_task(task_result['task_id'])
                for task_result in task_results
            ]
            
            results = await asyncio.gather(*execution_tasks)
            monitor.stop()
            
            # Performance assertions
            assert len(results) == performance_test_config['concurrent_operations']
            assert all(r['status'] == 'completed' for r in results)
            
            # Concurrent execution should be faster than sequential
            concurrent_throughput = len(results) / monitor.execution_time
            assert concurrent_throughput > 50  # Should handle concurrent load well
            
            print(f"Concurrent Task Execution: {concurrent_throughput:.2f} tasks/second")
    
    @pytest.mark.performance
    async def test_task_dependency_resolution_performance(self):
        """Test performance of dependency resolution with complex chains."""
        with MockServiceContext() as mocks:
            service = TaskService()
            await service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create dependency chain: Task 0 -> Task 1 -> Task 2 -> ... -> Task N
            chain_length = 20
            previous_task_id = None
            
            for i in range(chain_length):
                dependencies = [previous_task_id] if previous_task_id else []
                
                result = await service.create_task(
                    name=f"Chain Task {i}",
                    description=f"Task {i} in dependency chain",
                    dependencies=dependencies
                )
                
                if previous_task_id:
                    # Mark previous task as completed
                    service._active_tasks[previous_task_id]['status'] = 'completed'
                
                previous_task_id = result['task_id']
            
            # Execute final task (should resolve entire chain)
            await service.execute_task(previous_task_id, wait_for_dependencies=True)
            
            monitor.stop()
            
            # Dependency resolution should be efficient
            assert monitor.execution_time < 1.0  # Should resolve chain quickly
            
            print(f"Dependency chain resolution time: {monitor.execution_time:.3f}s for {chain_length} tasks")


class TestValidationServicePerformance:
    """Performance tests for Validation Service."""
    
    @pytest.mark.performance
    async def test_code_validation_throughput(self, performance_test_config):
        """Test validation service throughput with various code sizes."""
        with MockServiceContext() as mocks:
            service = ValidationService()
            await service.initialize()
            
            # Test different code sizes
            code_samples = [
                "def small(): pass",  # Small
                "def medium():\n" + "    x = 1\n" * 50 + "    return x",  # Medium
                "def large():\n" + "    x = 1\n" * 500 + "    return x"  # Large
            ]
            
            results = {}
            
            for size_name, code in zip(['small', 'medium', 'large'], code_samples):
                monitor = PerformanceMonitor()
                monitor.start()
                
                # Validate code multiple times
                tasks = [
                    service.validate_code(code, 'python')
                    for _ in range(performance_test_config['test_iterations'] // 10)
                ]
                
                validation_results = await asyncio.gather(*tasks)
                monitor.stop()
                
                results[size_name] = {
                    'validation_time': monitor.execution_time,
                    'throughput': len(validation_results) / monitor.execution_time,
                    'avg_time_per_validation': monitor.execution_time / len(validation_results)
                }
            
            # Validation should be fast for all code sizes
            assert all(r['avg_time_per_validation'] < 0.1 for r in results.values())  # <100ms each
            assert results['small']['throughput'] > 100  # >100 validations/second for small code
            
            print("Code validation performance:", results)
    
    @pytest.mark.performance
    async def test_placeholder_detection_performance(self, performance_test_config):
        """Test placeholder detection performance with various patterns."""
        with MockServiceContext() as mocks:
            service = ValidationService()
            await service.initialize()
            
            # Generate code with many placeholders
            placeholder_code = '''
def function_with_many_placeholders():
    # TODO: Implement this function
    pass  # implement later
    
    for i in range(100):
        # FIXME: Fix this loop
        if i % 2 == 0:
            ...  # placeholder
        else:
            # TODO: Add else logic
            raise NotImplementedError("Complete this")
    
    return None  # TODO: Return proper value
'''
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Run placeholder detection multiple times
            tasks = [
                service.validate_code(placeholder_code, 'python', check_placeholders=True)
                for _ in range(performance_test_config['test_iterations'] // 20)
            ]
            
            results = await asyncio.gather(*tasks)
            monitor.stop()
            
            # Performance assertions
            assert len(results) == performance_test_config['test_iterations'] // 20
            assert all(r['categories']['placeholder']['count'] > 0 for r in results)
            
            # Placeholder detection should be fast
            avg_time = monitor.execution_time / len(results)
            assert avg_time < 0.05  # <50ms per detection
            
            print(f"Placeholder detection - Average time: {avg_time:.3f}s per validation")
    
    @pytest.mark.performance
    async def test_validation_memory_efficiency(self, performance_test_config):
        """Test validation service memory efficiency."""
        with MockServiceContext() as mocks:
            service = ValidationService()
            await service.initialize()
            
            # Measure initial memory
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Perform many validations
            large_code = "def func():\n" + "    x = 1\n" * 1000  # Large code sample
            
            for i in range(50):  # Fewer iterations to avoid timeout
                await service.validate_code(large_code, 'python')
                
                # Periodic cleanup
                if i % 10 == 0:
                    gc.collect()
            
            # Measure final memory
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < performance_test_config['memory_limit_mb']
            
            print(f"Validation memory usage - Increase: {memory_increase:.2f}MB")


class TestIntegratedPerformance:
    """Performance tests for integrated service workflows."""
    
    @pytest.mark.performance
    async def test_full_workflow_performance(self, performance_test_config, temp_directory):
        """Test performance of complete service workflow."""
        with MockServiceContext() as mocks:
            # Initialize all services
            ai_service = AIService()
            await ai_service.initialize()
            
            project_service = ProjectService()
            await project_service.initialize()
            
            task_service = TaskService()
            task_service._ai_service = ai_service
            await task_service.initialize()
            
            validation_service = ValidationService()
            await validation_service.initialize()
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Execute complete workflow
            workflows = []
            for i in range(5):  # Limited iterations for complex workflow
                workflow = self._execute_complete_workflow(
                    i, ai_service, project_service, task_service, validation_service, temp_directory
                )
                workflows.append(workflow)
            
            results = await asyncio.gather(*workflows)
            monitor.stop()
            
            # Performance assertions
            assert len(results) == 5
            assert all(r['success'] for r in results)
            
            # Complete workflow should finish in reasonable time
            avg_workflow_time = monitor.execution_time / len(results)
            assert avg_workflow_time < 2.0  # <2 seconds per workflow
            
            print(f"Complete workflow - Average time: {avg_workflow_time:.3f}s per workflow")
    
    async def _execute_complete_workflow(
        self, 
        workflow_id: int,
        ai_service: AIService,
        project_service: ProjectService,
        task_service: TaskService,
        validation_service: ValidationService,
        temp_directory
    ) -> Dict[str, Any]:
        """Execute a complete development workflow."""
        try:
            # 1. Create project
            project_path = temp_directory / f"workflow_project_{workflow_id}"
            project_result = await project_service.create_project(
                name=f"Workflow Project {workflow_id}",
                path=project_path,
                project_type="python",
                initialize_git=False,
                create_venv=False
            )
            
            # 2. Generate code
            code_task = await task_service.create_task(
                name=f"Generate Code {workflow_id}",
                description="Generate utility functions",
                task_type="code_generation",
                ai_enabled=True
            )
            
            code_result = await task_service.execute_task(code_task['task_id'])
            
            # 3. Validate code
            generated_code = code_result['result']['data']['code']
            validation_result = await validation_service.validate_code(generated_code, 'python')
            
            # 4. Validate project
            project_validation = await project_service.validate_project(project_result['project_id'])
            
            return {
                'workflow_id': workflow_id,
                'success': True,
                'project_valid': project_validation['is_valid'],
                'code_valid': validation_result['is_valid'],
                'code_score': validation_result['score']
            }
            
        except Exception as e:
            return {
                'workflow_id': workflow_id,
                'success': False,
                'error': str(e)
            }
    
    @pytest.mark.performance
    async def test_service_registry_performance(self, performance_test_config):
        """Test service registry performance under load."""
        from services.base import ServiceRegistry
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Create multiple registries and services
        registries = []
        for i in range(10):
            registry = ServiceRegistry()
            
            with MockServiceContext():
                # Register services
                await registry.register_service(AIService, auto_initialize=False)
                await registry.register_service(ProjectService, auto_initialize=False)
                await registry.register_service(TaskService, auto_initialize=False)
                await registry.register_service(ValidationService, auto_initialize=False)
                
                # Health check all services
                await registry.health_check_all()
                
            registries.append(registry)
        
        monitor.stop()
        
        # Registry operations should be fast
        avg_time_per_registry = monitor.execution_time / len(registries)
        assert avg_time_per_registry < 0.1  # <100ms per registry setup
        
        print(f"Service registry setup - Average time: {avg_time_per_registry:.3f}s per registry")
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_stress_test_all_services(self, performance_test_config):
        """Stress test all services under heavy concurrent load."""
        with MockServiceContext() as mocks:
            # Configure mocks for stress testing
            mocks['task_engine'].set_execution_time(0.001)  # Very fast
            
            # Initialize services
            services = {
                'ai': AIService(),
                'project': ProjectService(), 
                'task': TaskService(),
                'validation': ValidationService()
            }
            
            for service in services.values():
                await service.initialize()
            
            services['task']._ai_service = services['ai']
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create massive concurrent load
            all_tasks = []
            
            # AI service tasks
            for i in range(50):
                task = services['ai'].generate_code(f"Function {i}", 'python')
                all_tasks.append(task)
            
            # Task service tasks
            for i in range(50):
                task_result = await services['task'].create_task(
                    f"Stress Task {i}", f"Description {i}"
                )
                task = services['task'].execute_task(task_result['task_id'])
                all_tasks.append(task)
            
            # Validation service tasks
            for i in range(50):
                task = services['validation'].validate_code(
                    f"def stress_func_{i}(): return {i}", 'python'
                )
                all_tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            monitor.stop()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            success_rate = len(successful_results) / len(results)
            
            # Should handle stress load reasonably well
            assert success_rate > 0.95  # >95% success rate
            assert monitor.execution_time < 10.0  # Complete within 10 seconds
            
            print(f"Stress test - Success rate: {success_rate:.2%}, "
                  f"Time: {monitor.execution_time:.3f}s, "
                  f"Failed: {len(failed_results)}")


if __name__ == "__main__":
    # Run performance tests manually
    import sys
    
    async def run_performance_tests():
        """Run performance tests manually."""
        print("Running performance tests...")
        
        # You can run individual test methods here for debugging
        # test_instance = TestAIServicePerformance()
        # await test_instance.test_code_generation_throughput({
        #     'test_iterations': 100,
        #     'memory_limit_mb': 100
        # })
        
        print("Performance tests completed.")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        asyncio.run(run_performance_tests())