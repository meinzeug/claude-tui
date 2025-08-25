"""
Service Integration Tests for claude-tui.

Tests interactions between services to ensure proper integration:
- AI Service + Task Service coordination
- Project Service + Validation Service workflows  
- Task Service + Validation Service quality checks
- Multi-service orchestration scenarios
- Cross-service dependency injection
- Performance under integrated workloads
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from services.ai_service import AIService
from services.project_service import ProjectService
from services.task_service import TaskService, TaskStatus
from services.validation_service import ValidationService
from services.base import ServiceRegistry, get_service_registry
from core.types import Priority


class TestServiceRegistryIntegration:
    """Test service registry and dependency injection integration."""
    
    @pytest.mark.integration
    async def test_service_registry_initialization(self):
        """Test service registry initialization with multiple services."""
        registry = ServiceRegistry()
        
        # Register services
        ai_service = await registry.register_service(AIService, auto_initialize=False)
        project_service = await registry.register_service(ProjectService, auto_initialize=False)
        task_service = await registry.register_service(TaskService, auto_initialize=False)
        validation_service = await registry.register_service(ValidationService, auto_initialize=False)
        
        assert isinstance(ai_service, AIService)
        assert isinstance(project_service, ProjectService)
        assert isinstance(task_service, TaskService)
        assert isinstance(validation_service, ValidationService)
        
        # Test service retrieval
        retrieved_ai = await registry.get_service(AIService)
        assert retrieved_ai is ai_service
    
    @pytest.mark.integration
    async def test_service_dependency_injection(self):
        """Test dependency injection between services."""
        registry = ServiceRegistry()
        
        # Mock dependencies to avoid external calls
        with patch('services.ai_service.AIInterface'):
            with patch('services.ai_service.ClaudeCodeIntegration'):
                with patch('services.ai_service.ClaudeFlowIntegration'):
                    with patch('services.project_service.ProjectManager'):
                        with patch('services.task_service.TaskEngine'):
                            with patch('services.validation_service.ProgressValidator'):
                                
                                ai_service = await registry.register_service(AIService)
                                task_service = await registry.register_service(TaskService)
                                
                                # Task service should have access to AI service
                                assert task_service._ai_service is not None
    
    @pytest.mark.integration
    async def test_service_health_checks(self):
        """Test health checks across all services."""
        registry = ServiceRegistry()
        
        with patch('services.ai_service.AIInterface'):
            with patch('services.ai_service.ClaudeCodeIntegration'):
                with patch('services.ai_service.ClaudeFlowIntegration'):
                    with patch('services.project_service.ProjectManager'):
                        with patch('services.task_service.TaskEngine'):
                            with patch('services.validation_service.ProgressValidator'):
                                
                                await registry.register_service(AIService)
                                await registry.register_service(ProjectService)
                                await registry.register_service(TaskService)
                                await registry.register_service(ValidationService)
                                
                                health_results = await registry.health_check_all()
                                
                                assert 'AIService' in health_results
                                assert 'ProjectService' in health_results
                                assert 'TaskService' in health_results
                                assert 'ValidationService' in health_results
                                
                                for service_name, health in health_results.items():
                                    assert health['status'] in ['healthy', 'unhealthy']


class TestAIServiceTaskServiceIntegration:
    """Test integration between AI Service and Task Service."""
    
    @pytest.mark.integration
    async def test_ai_assisted_task_execution(self):
        """Test task execution with AI assistance."""
        # Set up services with mocks
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.task_service.TaskEngine') as mock_task_engine:
                        
                        # Configure mocks
                        mock_ai_instance = AsyncMock()
                        mock_ai_interface.return_value = mock_ai_instance
                        mock_ai_instance.generate_code.return_value = {
                            'code': 'def generated_function(): return "AI generated"',
                            'metadata': {'model': 'claude-3'}
                        }
                        
                        mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                        mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                        mock_task_engine.return_value = AsyncMock()
                        
                        # Initialize services
                        ai_service = AIService()
                        await ai_service.initialize()
                        
                        task_service = TaskService()
                        task_service._ai_service = ai_service
                        await task_service.initialize()
                        
                        # Create and execute AI-assisted task
                        task_result = await task_service.create_task(
                            name="AI Code Generation",
                            description="Generate a utility function",
                            task_type="code_generation",
                            ai_enabled=True,
                            config={'language': 'python'}
                        )
                        
                        execution_result = await task_service.execute_task(task_result['task_id'])
                        
                        assert execution_result['status'] == TaskStatus.COMPLETED.value
                        assert execution_result['result']['type'] == 'ai_result'
                        assert 'AI generated' in execution_result['result']['data']['code']
                        
                        # Verify AI service was called
                        mock_ai_instance.generate_code.assert_called_once()
    
    @pytest.mark.integration
    async def test_task_orchestration_integration(self):
        """Test task orchestration using AI Service."""
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.task_service.TaskEngine') as mock_task_engine:
                        
                        # Configure AI Service mocks
                        mock_ai_instance = AsyncMock()
                        mock_ai_interface.return_value = mock_ai_instance
                        
                        mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                        mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                        
                        # Configure orchestration mock
                        mock_flow_instance = AsyncMock()
                        mock_claude_flow.return_value = mock_flow_instance
                        mock_flow_instance.orchestrate_task.return_value = {
                            'task_id': 'orchestrated-123',
                            'agents': ['coder', 'reviewer', 'tester'],
                            'status': 'running'
                        }
                        
                        mock_task_engine.return_value = AsyncMock()
                        
                        # Initialize services
                        ai_service = AIService()
                        await ai_service.initialize()
                        
                        task_service = TaskService()
                        task_service._ai_service = ai_service
                        await task_service.initialize()
                        
                        # Create orchestration task
                        task_result = await task_service.create_task(
                            name="Complex Workflow",
                            description="Build a complete application",
                            task_type="task_orchestration",
                            ai_enabled=True,
                            config={
                                'strategy': 'adaptive',
                                'requirements': {
                                    'components': ['backend', 'frontend', 'database'],
                                    'testing': True
                                }
                            }
                        )
                        
                        execution_result = await task_service.execute_task(task_result['task_id'])
                        
                        assert execution_result['status'] == TaskStatus.COMPLETED.value
                        assert execution_result['result']['data']['task_id'] == 'orchestrated-123'
                        assert len(execution_result['result']['data']['agents']) == 3
    
    @pytest.mark.integration
    async def test_ai_fallback_to_engine(self):
        """Test fallback from AI to engine when AI fails."""
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.task_service.TaskEngine') as mock_task_engine:
                        
                        # Configure AI Service to fail
                        mock_ai_instance = AsyncMock()
                        mock_ai_interface.return_value = mock_ai_instance
                        mock_ai_instance.generate_code.side_effect = Exception("AI service unavailable")
                        
                        mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                        mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                        
                        # Configure engine to succeed
                        mock_engine_instance = AsyncMock()
                        mock_task_engine.return_value = mock_engine_instance
                        mock_engine_instance.execute_task.return_value = {
                            'result': 'Engine fallback successful',
                            'method': 'task_engine'
                        }
                        
                        # Initialize services
                        ai_service = AIService()
                        await ai_service.initialize()
                        
                        task_service = TaskService()
                        task_service._ai_service = ai_service
                        await task_service.initialize()
                        
                        # Create task that will fallback to engine
                        task_result = await task_service.create_task(
                            name="Fallback Task",
                            description="Test AI fallback",
                            task_type="code_generation",
                            ai_enabled=True
                        )
                        
                        execution_result = await task_service.execute_task(task_result['task_id'])
                        
                        assert execution_result['status'] == TaskStatus.COMPLETED.value
                        assert execution_result['result']['type'] == 'engine_result'
                        assert 'fallback successful' in execution_result['result']['data']['result']


class TestProjectServiceValidationServiceIntegration:
    """Test integration between Project Service and Validation Service."""
    
    @pytest.mark.integration
    async def test_project_creation_with_validation(self, temp_directory):
        """Test project creation with integrated validation."""
        with patch('services.project_service.ProjectManager') as mock_project_manager:
            with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                
                # Configure project manager mock
                mock_pm_instance = AsyncMock()
                mock_project_manager.return_value = mock_pm_instance
                mock_pm_instance.create_project.return_value = {
                    'id': 'project-123',
                    'name': 'Validated Project',
                    'type': 'python'
                }
                mock_pm_instance.get_active_projects.return_value = []
                
                # Configure progress validator mock
                class MockProgressResult:
                    def __init__(self):
                        self.is_valid = True
                        self.authenticity_score = 0.9
                        self.issues = []
                        self.suggestions = ['Consider adding more tests']
                        self.placeholder_count = 0
                
                mock_pv_instance = AsyncMock()
                mock_progress_validator.return_value = mock_pv_instance
                mock_pv_instance.validate_progress.return_value = MockProgressResult()
                
                # Initialize services
                project_service = ProjectService()
                await project_service.initialize()
                
                validation_service = ValidationService()
                await validation_service.initialize()
                
                # Create project
                project_path = temp_directory / "validated_project"
                project_result = await project_service.create_project(
                    name="Validated Project",
                    path=project_path,
                    project_type="python",
                    initialize_git=False,
                    create_venv=False
                )
                
                # Validate project structure
                validation_result = await project_service.validate_project(
                    project_result['project_id']
                )
                
                assert validation_result['is_valid'] is True
                assert validation_result['score'] > 0.8
                
                # Validate specific file (if exists)
                if project_path.exists():
                    test_file = project_path / "main.py"
                    test_file.write_text("def main(): pass")
                    
                    progress_result = await validation_service.check_progress_authenticity(
                        test_file,
                        project_context={'project_id': project_result['project_id']}
                    )
                    
                    assert progress_result['is_authentic'] is True
                    assert progress_result['authenticity_score'] > 0.8
    
    @pytest.mark.integration
    async def test_project_loading_with_code_validation(self, temp_directory):
        """Test project loading with integrated code validation."""
        with patch('services.project_service.ProjectManager') as mock_project_manager:
            with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                
                # Create project directory and files
                project_path = temp_directory / "code_validation_project"
                project_path.mkdir()
                
                # Create sample Python files
                (project_path / "good_code.py").write_text('''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''')
                
                (project_path / "bad_code.py").write_text('''
def incomplete_function():
    # TODO: Implement this
    pass
    
def broken_syntax(
    # Missing closing parenthesis
    return "broken"
''')
                
                # Configure mocks
                mock_pm_instance = AsyncMock()
                mock_project_manager.return_value = mock_pm_instance
                mock_pm_instance.load_project.return_value = {
                    'name': 'Code Validation Project',
                    'type': 'python'
                }
                mock_pm_instance.get_active_projects.return_value = []
                
                mock_pv_instance = AsyncMock()
                mock_progress_validator.return_value = mock_pv_instance
                
                # Initialize services
                project_service = ProjectService()
                await project_service.initialize()
                
                validation_service = ValidationService()
                await validation_service.initialize()
                
                # Load project
                project_result = await project_service.load_project(project_path)
                
                # Validate good code
                good_code = (project_path / "good_code.py").read_text()
                good_validation = await validation_service.validate_code(good_code, 'python')
                
                assert good_validation['is_valid'] is True
                assert good_validation['score'] > 0.8
                
                # Validate bad code
                bad_code = (project_path / "bad_code.py").read_text()
                bad_validation = await validation_service.validate_code(bad_code, 'python')
                
                assert bad_validation['is_valid'] is False
                assert 'placeholder' in bad_validation['categories']
                assert bad_validation['categories']['placeholder']['count'] > 0


class TestTaskServiceValidationServiceIntegration:
    """Test integration between Task Service and Validation Service."""
    
    @pytest.mark.integration
    async def test_task_execution_with_code_validation(self):
        """Test task execution with integrated code validation."""
        with patch('services.task_service.TaskEngine') as mock_task_engine:
            with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                
                # Configure task engine mock
                mock_engine_instance = AsyncMock()
                mock_task_engine.return_value = mock_engine_instance
                mock_engine_instance.execute_task.return_value = {
                    'code': 'def validated_function(): return "success"',
                    'language': 'python'
                }
                
                # Configure progress validator mock
                mock_pv_instance = AsyncMock()
                mock_progress_validator.return_value = mock_pv_instance
                
                # Initialize services
                task_service = TaskService()
                await task_service.initialize()
                
                validation_service = ValidationService()
                await validation_service.initialize()
                
                # Create code generation task
                task_result = await task_service.create_task(
                    name="Validated Code Generation",
                    description="Generate code with validation",
                    task_type="code_generation",
                    ai_enabled=False,
                    config={'language': 'python', 'validate_output': True}
                )
                
                # Execute task
                execution_result = await task_service.execute_task(task_result['task_id'])
                
                # Validate the generated code
                generated_code = execution_result['result']['data']['code']
                validation_result = await validation_service.validate_code(
                    generated_code,
                    'python'
                )
                
                assert execution_result['status'] == TaskStatus.COMPLETED.value
                assert validation_result['is_valid'] is True
                assert validation_result['score'] > 0.8
    
    @pytest.mark.integration
    async def test_task_quality_assurance_workflow(self):
        """Test complete task quality assurance workflow."""
        with patch('services.task_service.TaskEngine') as mock_task_engine:
            with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                
                # Configure mocks
                mock_engine_instance = AsyncMock()
                mock_task_engine.return_value = mock_engine_instance
                
                mock_pv_instance = AsyncMock()
                mock_progress_validator.return_value = mock_pv_instance
                
                # Initialize services
                task_service = TaskService()
                await task_service.initialize()
                
                validation_service = ValidationService()
                await validation_service.initialize()
                
                # Simulate QA workflow: Create -> Execute -> Validate -> Report
                
                # 1. Create multiple tasks
                tasks = []
                for i in range(3):
                    mock_engine_instance.execute_task.return_value = {
                        'code': f'def function_{i}(): return {i}',
                        'metadata': {'task_id': i}
                    }
                    
                    task_result = await task_service.create_task(
                        name=f"QA Task {i}",
                        description=f"Task {i} for QA workflow",
                        task_type="code_generation"
                    )
                    tasks.append(task_result)
                
                # 2. Execute tasks
                execution_results = []
                for task in tasks:
                    result = await task_service.execute_task(task['task_id'])
                    execution_results.append(result)
                
                # 3. Validate all generated code
                validation_results = []
                for result in execution_results:
                    if 'code' in result['result']['data']:
                        validation = await validation_service.validate_code(
                            result['result']['data']['code'],
                            'python'
                        )
                        validation_results.append(validation)
                
                # 4. Generate quality report
                task_performance = await task_service.get_performance_report()
                validation_report = await validation_service.get_validation_report()
                
                # Assertions
                assert len(execution_results) == 3
                assert all(r['status'] == TaskStatus.COMPLETED.value for r in execution_results)
                assert len(validation_results) == 3
                assert all(v['is_valid'] for v in validation_results)
                
                assert task_performance['success_rate'] == 1.0
                assert validation_report['success_rate'] > 0.0


class TestMultiServiceOrchestrationScenarios:
    """Test complex multi-service orchestration scenarios."""
    
    @pytest.mark.integration
    async def test_full_development_workflow(self, temp_directory):
        """Test complete development workflow using all services."""
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.project_service.ProjectManager') as mock_project_manager:
                        with patch('services.task_service.TaskEngine') as mock_task_engine:
                            with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                                
                                # Configure all mocks
                                mock_ai_instance = AsyncMock()
                                mock_ai_interface.return_value = mock_ai_instance
                                mock_ai_instance.generate_code.return_value = {
                                    'code': '''
def create_user(name, email):
    """Create a new user with validation."""
    if not name or not email:
        raise ValueError("Name and email are required")
    return {"name": name, "email": email, "id": generate_id()}

def generate_id():
    """Generate unique user ID."""
    import uuid
    return str(uuid.uuid4())
''',
                                    'metadata': {'model': 'claude-3', 'validated': True}
                                }
                                
                                mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                                mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                                
                                mock_pm_instance = AsyncMock()
                                mock_project_manager.return_value = mock_pm_instance
                                mock_pm_instance.create_project.return_value = {
                                    'id': 'workflow-project',
                                    'name': 'Workflow Project',
                                    'type': 'python'
                                }
                                mock_pm_instance.get_active_projects.return_value = []
                                
                                mock_engine_instance = AsyncMock()
                                mock_task_engine.return_value = mock_engine_instance
                                mock_engine_instance.execute_task.return_value = {'result': 'success'}
                                
                                class MockProgressResult:
                                    def __init__(self):
                                        self.is_valid = True
                                        self.authenticity_score = 0.95
                                        self.issues = []
                                        self.suggestions = []
                                        self.placeholder_count = 0
                                
                                mock_pv_instance = AsyncMock()
                                mock_progress_validator.return_value = mock_pv_instance
                                mock_pv_instance.validate_progress.return_value = MockProgressResult()
                                
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
                                
                                # WORKFLOW: Project Creation -> Code Generation -> Validation -> Quality Check
                                
                                # 1. Create Project
                                project_path = temp_directory / "workflow_project"
                                project_result = await project_service.create_project(
                                    name="Workflow Project",
                                    path=project_path,
                                    project_type="python"
                                )
                                
                                # 2. Generate Code using AI-assisted Task
                                code_task = await task_service.create_task(
                                    name="Generate User Management Code",
                                    description="Create user management functions",
                                    task_type="code_generation",
                                    ai_enabled=True,
                                    config={'language': 'python', 'domain': 'user_management'}
                                )
                                
                                code_execution = await task_service.execute_task(code_task['task_id'])
                                generated_code = code_execution['result']['data']['code']
                                
                                # 3. Validate Generated Code
                                code_validation = await validation_service.validate_code(
                                    generated_code,
                                    'python',
                                    validation_level='comprehensive'
                                )
                                
                                # 4. Save Code to Project and Validate Progress
                                code_file = project_path / "user_management.py"
                                if project_path.exists():
                                    code_file.write_text(generated_code)
                                    
                                    progress_validation = await validation_service.check_progress_authenticity(
                                        code_file,
                                        project_context={'project_id': project_result['project_id']}
                                    )
                                
                                # 5. Generate Reports
                                project_validation = await project_service.validate_project(
                                    project_result['project_id']
                                )
                                task_report = await task_service.get_performance_report()
                                validation_report = await validation_service.get_validation_report()
                                
                                # Verify complete workflow
                                assert project_result['name'] == "Workflow Project"
                                assert code_execution['status'] == TaskStatus.COMPLETED.value
                                assert code_validation['is_valid'] is True
                                assert code_validation['score'] > 0.8
                                assert task_report['success_rate'] == 1.0
                                assert validation_report['success_rate'] > 0.0
                                
                                # Verify code quality
                                assert 'def create_user' in generated_code
                                assert 'def generate_id' in generated_code
                                assert len(code_validation['issues']) == 0
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_multi_service_operations(self, performance_test_config):
        """Test concurrent operations across multiple services."""
        from tests.conftest import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.task_service.TaskEngine') as mock_task_engine:
                        with patch('services.validation_service.ProgressValidator') as mock_progress_validator:
                            
                            # Configure mocks for fast responses
                            mock_ai_instance = AsyncMock()
                            mock_ai_interface.return_value = mock_ai_instance
                            mock_ai_instance.generate_code.return_value = {
                                'code': 'def concurrent_function(): pass',
                                'metadata': {}
                            }
                            
                            mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                            mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                            
                            mock_engine_instance = AsyncMock()
                            mock_task_engine.return_value = mock_engine_instance
                            mock_engine_instance.execute_task.return_value = {'result': 'concurrent_success'}
                            
                            mock_pv_instance = AsyncMock()
                            mock_progress_validator.return_value = mock_pv_instance
                            
                            # Initialize services
                            ai_service = AIService()
                            await ai_service.initialize()
                            
                            task_service = TaskService()
                            task_service._ai_service = ai_service
                            await task_service.initialize()
                            
                            validation_service = ValidationService()
                            await validation_service.initialize()
                            
                            monitor.start()
                            
                            # Create concurrent operations
                            concurrent_operations = []
                            
                            for i in range(performance_test_config['concurrent_operations']):
                                # Mix of different service operations
                                if i % 3 == 0:
                                    # AI code generation
                                    op = ai_service.generate_code(f"Generate function {i}", 'python')
                                elif i % 3 == 1:
                                    # Task execution
                                    task_result = await task_service.create_task(f"Task {i}", f"Description {i}")
                                    op = task_service.execute_task(task_result['task_id'])
                                else:
                                    # Code validation
                                    op = validation_service.validate_code(f"def func_{i}(): pass", 'python')
                                
                                concurrent_operations.append(op)
                            
                            # Execute all operations concurrently
                            results = await asyncio.gather(*concurrent_operations)
                            
                            monitor.stop()
                            
                            # Verify results
                            assert len(results) == performance_test_config['concurrent_operations']
                            
                            # Performance assertion
                            monitor.assert_performance(performance_test_config['max_execution_time'])
    
    @pytest.mark.integration
    async def test_error_propagation_across_services(self):
        """Test error propagation and handling across services."""
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    with patch('services.task_service.TaskEngine') as mock_task_engine:
                        
                        # Configure AI service to fail
                        mock_ai_instance = AsyncMock()
                        mock_ai_interface.return_value = mock_ai_instance
                        mock_ai_instance.generate_code.side_effect = Exception("AI service error")
                        
                        mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                        mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                        
                        # Configure task engine for fallback
                        mock_engine_instance = AsyncMock()
                        mock_task_engine.return_value = mock_engine_instance
                        mock_engine_instance.execute_task.return_value = {'result': 'fallback_success'}
                        
                        # Initialize services
                        ai_service = AIService()
                        await ai_service.initialize()
                        
                        task_service = TaskService()
                        task_service._ai_service = ai_service
                        await task_service.initialize()
                        
                        # Create task that will trigger AI failure and fallback
                        task_result = await task_service.create_task(
                            name="Error Propagation Test",
                            description="Test error handling",
                            task_type="code_generation",
                            ai_enabled=True
                        )
                        
                        # Execute task (should fallback to engine)
                        execution_result = await task_service.execute_task(task_result['task_id'])
                        
                        # Verify graceful error handling and fallback
                        assert execution_result['status'] == TaskStatus.COMPLETED.value
                        assert execution_result['result']['type'] == 'engine_result'
                        assert 'fallback_success' in execution_result['result']['data']['result']