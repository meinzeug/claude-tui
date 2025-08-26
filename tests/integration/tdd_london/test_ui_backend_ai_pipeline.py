"""
TDD London School Integration Tests for UI-Backend-AI Pipeline

London School integration testing approach:
- Test complete workflow interactions between UI, Backend, and AI components
- Mock external dependencies while testing real object collaborations
- Verify behavior and contracts across system boundaries
- Focus on message passing and coordination patterns
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

from src.claude_tui.ui.main_app import ClaudeTUIApp
from src.claude_tui.core.project_manager import ProjectManager
from src.claude_tui.integrations.ai_interface import AIInterface
from src.ai.swarm_orchestrator import SwarmOrchestrator
from src.api.main import app as api_app
from src.database.service import DatabaseService


# Mock Infrastructure - London School external dependency isolation
@pytest.fixture
def mock_database_service():
    """Mock database service for integration testing"""
    mock = AsyncMock(spec=DatabaseService)
    mock.create_project = AsyncMock(return_value={"id": "proj-123", "status": "created"})
    mock.get_project = AsyncMock(return_value={"id": "proj-123", "name": "test-project"})
    mock.update_project = AsyncMock(return_value={"id": "proj-123", "status": "updated"})
    mock.save_task_result = AsyncMock()
    mock.get_project_tasks = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_external_ai_service():
    """Mock external AI service calls (Claude API, etc.)"""
    mock = AsyncMock()
    mock.generate_code = AsyncMock(return_value="def hello(): return 'Hello, World!'")
    mock.analyze_requirements = AsyncMock(return_value={
        "tasks": [
            {"name": "implement_auth", "priority": "high"},
            {"name": "create_api", "priority": "medium"}
        ],
        "complexity": 7
    })
    mock.validate_code = AsyncMock(return_value={"valid": True, "issues": []})
    return mock


@pytest.fixture
def integrated_system_with_mocks(mock_database_service, mock_external_ai_service):
    """Integrated system with external dependencies mocked"""
    
    # Create real components that will interact
    project_manager = ProjectManager(
        config_manager=Mock(),
        state_manager=Mock(),
        task_engine=Mock(), 
        ai_interface=Mock(),
        validator=Mock()
    )
    
    swarm_orchestrator = SwarmOrchestrator(
        max_swarms=2,
        max_agents_per_swarm=3,
        enable_auto_scaling=False  # Disable for testing
    )
    
    # Mock external services
    with patch('src.database.service.DatabaseService', return_value=mock_database_service), \
         patch('src.claude_tui.integrations.ai_interface.external_ai_client', mock_external_ai_service):
        
        return {
            'project_manager': project_manager,
            'swarm_orchestrator': swarm_orchestrator,
            'database': mock_database_service,
            'ai_service': mock_external_ai_service
        }


class TestUIToBackendIntegration:
    """Test UI to Backend communication patterns - London School interaction focus"""
    
    @pytest.mark.asyncio
    async def test_project_creation_workflow_from_ui_to_backend(
        self,
        integrated_system_with_mocks
    ):
        """Test complete project creation workflow from UI trigger to backend storage"""
        
        # Arrange
        project_manager = integrated_system_with_mocks['project_manager']
        database = integrated_system_with_mocks['database']
        
        project_data = {
            "name": "new-react-app",
            "template": "react-typescript",
            "description": "Modern React application with TypeScript"
        }
        
        # Mock UI interaction trigger
        ui_request = Mock()
        ui_request.get_project_data.return_value = project_data
        
        # Mock project manager workflow
        with patch.object(project_manager, 'create_project') as mock_create, \
             patch.object(project_manager, 'save_project') as mock_save:
            
            mock_project = Mock()
            mock_project.name = project_data["name"]
            mock_project.to_dict = Mock(return_value=project_data)
            mock_create.return_value = mock_project
            
            # Act - Simulate UI triggering project creation
            created_project = await project_manager.create_project(
                template_name=project_data["template"],
                project_name=project_data["name"], 
                output_directory=Path("/tmp/projects")
            )
            
            # Simulate saving to backend
            await project_manager.save_project(created_project)
        
        # Assert - Verify complete UI->Backend workflow
        mock_create.assert_called_once()
        mock_save.assert_called_once_with(created_project)
    
    @pytest.mark.asyncio
    async def test_task_execution_request_from_ui_to_backend(
        self,
        integrated_system_with_mocks
    ):
        """Test task execution workflow initiated from UI"""
        
        # Arrange
        project_manager = integrated_system_with_mocks['project_manager']
        
        task_request = {
            "description": "Add user authentication to the application",
            "priority": "high",
            "requirements": {
                "features": ["login", "register", "logout"],
                "security": "JWT tokens"
            }
        }
        
        # Mock UI task submission
        ui_task_widget = Mock()
        ui_task_widget.get_task_data.return_value = task_request
        
        # Mock backend task processing
        with patch.object(project_manager, 'orchestrate_development') as mock_orchestrate:
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.completed_tasks = ["implement_auth", "add_tests"]
            mock_orchestrate.return_value = mock_result
            
            # Act - Simulate UI submitting task
            result = await project_manager.orchestrate_development(task_request)
        
        # Assert - Verify task processing workflow
        mock_orchestrate.assert_called_once_with(task_request)
        assert result.success is True


class TestBackendToAIIntegration:
    """Test Backend to AI service integration - London School contract verification"""
    
    @pytest.mark.asyncio
    async def test_swarm_orchestration_with_ai_backend_coordination(
        self,
        integrated_system_with_mocks
    ):
        """Test swarm orchestrator coordinating with AI backend services"""
        
        # Arrange
        swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
        ai_service = integrated_system_with_mocks['ai_service']
        
        # Mock swarm initialization and AI coordination
        with patch.object(swarm_orchestrator, 'initialize_swarm') as mock_init_swarm, \
             patch.object(swarm_orchestrator, 'execute_task') as mock_execute_task:
            
            mock_init_swarm.return_value = "swarm-ai-integration"
            mock_execute_task.return_value = "exec-ai-123"
            
            project_spec = {
                "description": "Build AI-powered recommendation system",
                "features": ["ml_models", "data_processing", "api_endpoints"],
                "complexity": 8
            }
            
            # Act - Initialize swarm and execute AI task
            swarm_id = await swarm_orchestrator.initialize_swarm(project_spec)
            
            from src.ai.swarm_orchestrator import TaskRequest
            ai_task = TaskRequest(
                task_id="ai-recommendation-task",
                description="Implement ML recommendation algorithm",
                agent_requirements=["ml_developer", "data_scientist"],
                estimated_complexity=8
            )
            
            execution_id = await swarm_orchestrator.execute_task(ai_task, swarm_id)
        
        # Assert - Verify AI backend coordination
        mock_init_swarm.assert_called_once_with(project_spec)
        mock_execute_task.assert_called_once()
        assert swarm_id == "swarm-ai-integration"
        assert execution_id == "exec-ai-123"
    
    @pytest.mark.asyncio
    async def test_ai_code_generation_and_backend_validation_pipeline(
        self,
        integrated_system_with_mocks
    ):
        """Test AI code generation followed by backend validation"""
        
        # Arrange
        project_manager = integrated_system_with_mocks['project_manager']
        ai_service = integrated_system_with_mocks['ai_service']
        
        code_request = {
            "function_name": "authenticate_user",
            "parameters": ["username", "password"],
            "return_type": "AuthResult",
            "requirements": "Validate user credentials and return JWT token"
        }
        
        # Mock AI code generation
        generated_code = """
def authenticate_user(username: str, password: str) -> AuthResult:
    # Validate credentials
    user = User.find_by_username(username)
    if user and user.verify_password(password):
        token = jwt.encode({'user_id': user.id}, SECRET_KEY)
        return AuthResult(success=True, token=token)
    return AuthResult(success=False, error="Invalid credentials")
"""
        
        ai_service.generate_code.return_value = generated_code
        ai_service.validate_code.return_value = {"valid": True, "issues": []}
        
        # Mock backend validation workflow
        with patch.object(project_manager, 'validator') as mock_validator:
            
            validation_result = Mock()
            validation_result.is_valid = True
            validation_result.issues = []
            mock_validator.validate_generated_code = AsyncMock(return_value=validation_result)
            
            # Act - Generate code and validate
            code = await ai_service.generate_code(code_request)
            ai_validation = await ai_service.validate_code(code)
            backend_validation = await mock_validator.validate_generated_code(code)
        
        # Assert - Verify complete AI->Backend validation pipeline
        ai_service.generate_code.assert_called_once_with(code_request)
        ai_service.validate_code.assert_called_once_with(generated_code)
        mock_validator.validate_generated_code.assert_called_once_with(generated_code)
        
        assert code == generated_code
        assert ai_validation["valid"] is True
        assert backend_validation.is_valid is True


class TestUIToAIDirectIntegration:
    """Test direct UI to AI interactions - London School end-to-end behavior"""
    
    @pytest.mark.asyncio
    async def test_real_time_ai_assistance_in_ui(
        self,
        integrated_system_with_mocks
    ):
        """Test real-time AI assistance triggered from UI components"""
        
        # Arrange
        ai_service = integrated_system_with_mocks['ai_service']
        
        # Mock UI code editor requesting AI assistance
        ui_code_editor = Mock()
        ui_code_editor.get_current_context.return_value = {
            "current_code": "def calculate_",
            "cursor_position": 15,
            "file_type": "python",
            "project_context": "data analysis tool"
        }
        
        # Mock AI code completion
        ai_service.complete_code = AsyncMock(return_value={
            "completions": [
                "def calculate_mean(data): return sum(data) / len(data)",
                "def calculate_median(data): return sorted(data)[len(data)//2]",
                "def calculate_mode(data): return max(set(data), key=data.count)"
            ],
            "confidence": 0.92
        })
        
        # Act - Simulate UI requesting AI code completion
        context = ui_code_editor.get_current_context()
        completions = await ai_service.complete_code(context)
        
        # Assert - Verify UI->AI interaction
        ai_service.complete_code.assert_called_once_with(context)
        assert len(completions["completions"]) == 3
        assert completions["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_ui_project_analysis_with_ai_insights(
        self,
        integrated_system_with_mocks
    ):
        """Test UI project analysis dashboard with AI-powered insights"""
        
        # Arrange
        ai_service = integrated_system_with_mocks['ai_service']
        project_manager = integrated_system_with_mocks['project_manager']
        
        # Mock UI project dashboard requesting analysis
        ui_dashboard = Mock()
        ui_dashboard.get_project_files.return_value = [
            "src/main.py", "src/utils.py", "tests/test_main.py",
            "requirements.txt", "README.md"
        ]
        
        # Mock AI project analysis
        ai_service.analyze_project = AsyncMock(return_value={
            "code_quality": {"score": 8.5, "issues": ["Missing docstrings", "Long functions"]},
            "test_coverage": {"percentage": 78, "missing_tests": ["utils.py"]},
            "dependencies": {"outdated": ["requests==2.25.1"], "vulnerabilities": []},
            "suggestions": [
                "Add type hints to improve code readability",
                "Increase test coverage for utils module",
                "Update requests library for security fixes"
            ]
        })
        
        # Mock project manager getting project status
        with patch.object(project_manager, 'get_project_status') as mock_status:
            
            mock_status.return_value = {
                "project": {"name": "test-project"},
                "validation": {"status": "valid"},
                "filesystem": {"files": 15, "size": "2.3MB"}
            }
            
            # Act - Simulate UI dashboard requesting analysis
            project_files = ui_dashboard.get_project_files()
            ai_analysis = await ai_service.analyze_project(project_files)
            project_status = await project_manager.get_project_status()
        
        # Assert - Verify UI->AI->Backend analysis workflow
        ai_service.analyze_project.assert_called_once_with(project_files)
        mock_status.assert_called_once()
        
        assert ai_analysis["code_quality"]["score"] == 8.5
        assert ai_analysis["test_coverage"]["percentage"] == 78
        assert len(ai_analysis["suggestions"]) == 3
        assert project_status["project"]["name"] == "test-project"


class TestFullPipelineIntegration:
    """Test complete pipeline integration - London School system-wide behavior"""
    
    @pytest.mark.asyncio
    async def test_complete_feature_development_pipeline(
        self,
        integrated_system_with_mocks
    ):
        """Test complete feature development from UI request to AI implementation to backend storage"""
        
        # Arrange
        project_manager = integrated_system_with_mocks['project_manager']
        swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
        ai_service = integrated_system_with_mocks['ai_service']
        database = integrated_system_with_mocks['database']
        
        # Feature request from UI
        feature_request = {
            "name": "user_profile_management",
            "description": "Allow users to create and manage their profiles",
            "requirements": [
                "Profile creation form",
                "Profile editing functionality", 
                "Profile picture upload",
                "Privacy settings"
            ],
            "priority": "high"
        }
        
        # Mock complete pipeline workflow
        with patch.object(project_manager, 'orchestrate_development') as mock_orchestrate, \
             patch.object(swarm_orchestrator, 'execute_task') as mock_swarm_execute, \
             patch.object(project_manager, 'save_project') as mock_save:
            
            # Mock AI-generated implementation
            mock_implementation_result = Mock()
            mock_implementation_result.success = True
            mock_implementation_result.generated_files = [
                "src/components/UserProfile.tsx",
                "src/api/profile.py", 
                "tests/test_profile.py"
            ]
            mock_implementation_result.validation_results = {"status": "valid"}
            
            mock_orchestrate.return_value = mock_implementation_result
            mock_swarm_execute.return_value = "exec-feature-123"
            
            # Act - Execute complete pipeline
            
            # Step 1: UI triggers feature development
            development_result = await project_manager.orchestrate_development(feature_request)
            
            # Step 2: Swarm executes AI-powered implementation
            from src.ai.swarm_orchestrator import TaskRequest
            swarm_task = TaskRequest(
                task_id="profile-feature-task",
                description=feature_request["description"],
                agent_requirements=["frontend", "backend", "tester"],
                estimated_complexity=7
            )
            execution_id = await swarm_orchestrator.execute_task(swarm_task)
            
            # Step 3: Save results to backend
            await project_manager.save_project()
        
        # Assert - Verify complete pipeline execution
        
        # Verify development orchestration
        mock_orchestrate.assert_called_once_with(feature_request)
        
        # Verify swarm execution
        mock_swarm_execute.assert_called_once()
        
        # Verify backend persistence
        mock_save.assert_called_once()
        
        # Verify pipeline results
        assert development_result.success is True
        assert len(development_result.generated_files) == 3
        assert execution_id == "exec-feature-123"
    
    @pytest.mark.asyncio
    async def test_error_recovery_across_pipeline_components(
        self,
        integrated_system_with_mocks
    ):
        """Test error recovery and coordination across pipeline components"""
        
        # Arrange
        project_manager = integrated_system_with_mocks['project_manager']
        ai_service = integrated_system_with_mocks['ai_service']
        database = integrated_system_with_mocks['database']
        
        # Simulate AI service failure
        ai_service.generate_code.side_effect = Exception("AI service temporarily unavailable")
        
        # Mock error recovery mechanisms
        with patch.object(project_manager, 'orchestrate_development') as mock_orchestrate:
            
            # Mock fallback implementation
            fallback_result = Mock()
            fallback_result.success = False
            fallback_result.error_message = "AI service failed, using template fallback"
            fallback_result.fallback_used = True
            
            mock_orchestrate.side_effect = [
                Exception("AI service failed"),  # First attempt fails
                fallback_result  # Second attempt with fallback succeeds
            ]
            
            # Act - Trigger development with error recovery
            try:
                # First attempt - should fail
                await project_manager.orchestrate_development({"feature": "test"})
                assert False, "Should have raised exception"
            except Exception as e:
                assert "AI service failed" in str(e)
            
            # Second attempt - with fallback
            result = await project_manager.orchestrate_development({"feature": "test", "use_fallback": True})
        
        # Assert - Verify error recovery workflow
        assert mock_orchestrate.call_count == 2
        assert result.success is False  # Failed but handled gracefully
        assert result.fallback_used is True
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_operations_coordination(
        self,
        integrated_system_with_mocks
    ):
        """Test coordination of concurrent pipeline operations"""
        
        # Arrange
        swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(3):
            from src.ai.swarm_orchestrator import TaskRequest
            task = TaskRequest(
                task_id=f"concurrent-task-{i}",
                description=f"Implement feature {i}",
                priority="medium",
                estimated_complexity=5
            )
            tasks.append(task)
        
        # Mock concurrent execution
        with patch.object(swarm_orchestrator, 'execute_task') as mock_execute:
            
            # Mock different execution IDs for each task
            mock_execute.side_effect = [
                f"exec-concurrent-{i}" for i in range(3)
            ]
            
            # Act - Execute tasks concurrently
            execution_ids = await asyncio.gather(*[
                swarm_orchestrator.execute_task(task) for task in tasks
            ])
        
        # Assert - Verify concurrent execution coordination
        assert len(execution_ids) == 3
        assert mock_execute.call_count == 3
        
        # Verify unique execution IDs
        assert len(set(execution_ids)) == 3
        assert all(exec_id.startswith("exec-concurrent-") for exec_id in execution_ids)


class TestPerformancePipelineIntegration:
    """Performance testing for integrated pipeline - London School benchmarking"""
    
    def test_pipeline_initialization_performance(
        self,
        benchmark,
        integrated_system_with_mocks
    ):
        """Benchmark complete pipeline initialization time"""
        
        def initialize_pipeline():
            project_manager = integrated_system_with_mocks['project_manager']
            swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
            return {
                'project_manager': project_manager,
                'swarm_orchestrator': swarm_orchestrator
            }
        
        result = benchmark(initialize_pipeline)
        assert result is not None
        assert 'project_manager' in result
        assert 'swarm_orchestrator' in result
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing_performance(
        self,
        integrated_system_with_mocks
    ):
        """Test performance of concurrent task processing in pipeline"""
        
        # Arrange
        swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
        
        # Create performance test tasks
        from src.ai.swarm_orchestrator import TaskRequest
        performance_tasks = [
            TaskRequest(
                task_id=f"perf-task-{i}",
                description=f"Performance test task {i}",
                estimated_complexity=3
            ) for i in range(10)
        ]
        
        # Mock fast execution
        with patch.object(swarm_orchestrator, 'execute_task') as mock_execute:
            mock_execute.side_effect = [f"exec-perf-{i}" for i in range(10)]
            
            # Act - Measure concurrent execution time
            start_time = datetime.utcnow()
            execution_ids = await asyncio.gather(*[
                swarm_orchestrator.execute_task(task) for task in performance_tasks
            ])
            end_time = datetime.utcnow()
        
        # Assert - Verify performance characteristics
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 1.0  # Should complete within 1 second with mocks
        assert len(execution_ids) == 10
        assert mock_execute.call_count == 10


class TestPipelineContractCompliance:
    """Test pipeline contract compliance - London School integration contracts"""
    
    def test_pipeline_components_maintain_expected_interfaces(
        self,
        integrated_system_with_mocks
    ):
        """Verify all pipeline components maintain expected interfaces"""
        
        project_manager = integrated_system_with_mocks['project_manager']
        swarm_orchestrator = integrated_system_with_mocks['swarm_orchestrator']
        
        # Verify ProjectManager interface
        assert hasattr(project_manager, 'create_project')
        assert hasattr(project_manager, 'orchestrate_development')
        assert hasattr(project_manager, 'get_project_status')
        
        # Verify SwarmOrchestrator interface  
        assert hasattr(swarm_orchestrator, 'initialize_swarm')
        assert hasattr(swarm_orchestrator, 'execute_task')
        assert hasattr(swarm_orchestrator, 'get_swarm_status')
        
        # Verify interfaces are callable
        assert callable(getattr(project_manager, 'orchestrate_development'))
        assert callable(getattr(swarm_orchestrator, 'execute_task'))
    
    def test_pipeline_error_handling_contracts(
        self,
        integrated_system_with_mocks
    ):
        """Verify pipeline components handle errors according to contracts"""
        
        project_manager = integrated_system_with_mocks['project_manager']
        
        # Test error handling contract
        with patch.object(project_manager, 'orchestrate_development') as mock_orchestrate:
            
            # Mock error response following contract
            from src.claude_tui.core.project_manager import DevelopmentResult
            error_result = DevelopmentResult(
                success=False,
                project=Mock(),
                completed_tasks=[],
                failed_tasks=[Mock()],
                validation_results={},
                duration=0.0,
                error_message="Development failed due to validation errors"
            )
            
            mock_orchestrate.return_value = error_result
            
            # Verify error result follows contract
            result = mock_orchestrate.return_value
            assert hasattr(result, 'success')
            assert hasattr(result, 'error_message')
            assert result.success is False
            assert result.error_message is not None