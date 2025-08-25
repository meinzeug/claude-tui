"""Comprehensive tests for AI Interface with Anti-Hallucination Engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from claude_tiu.integrations.ai_interface import AIInterface
from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.task import DevelopmentTask, TaskResult, TaskType, TaskPriority
from claude_tiu.models.project import Project
from claude_tiu.models.ai_models import AIRequest, AIResponse, CodeResult, WorkflowRequest
from claude_tiu.validation.progress_validator import ValidationResult, ValidationSeverity


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    manager = Mock(spec=ConfigManager)
    manager.get_setting = AsyncMock(return_value='test_value')
    return manager


@pytest.fixture
def mock_claude_code_client():
    """Mock Claude Code client."""
    client = Mock()
    client.execute_coding_task = AsyncMock()
    client.cleanup = AsyncMock()
    return client


@pytest.fixture
def mock_claude_flow_client():
    """Mock Claude Flow client."""
    client = Mock()
    client.execute_workflow = AsyncMock()
    client.cleanup = AsyncMock()
    return client


@pytest.fixture
def mock_anti_hallucination():
    """Mock Anti-Hallucination Engine."""
    engine = Mock()
    engine.initialize = AsyncMock()
    engine.cleanup = AsyncMock()
    engine.register_ai_interface_hooks = AsyncMock()
    engine.validate_ai_generated_content = AsyncMock()
    engine.validate_task_result = AsyncMock()
    engine.auto_fix_issues = AsyncMock(return_value=(False, None))
    engine.get_integration_metrics = AsyncMock(return_value={})
    engine.validate_project_codebase = AsyncMock(return_value={})
    engine.retrain_models = AsyncMock(return_value={})
    return engine


@pytest.fixture
def mock_context_builder():
    """Mock context builder."""
    builder = Mock()
    builder.build_smart_context = AsyncMock(return_value={'context': 'built'})
    builder.build_project_context = AsyncMock(return_value={'project': 'context'})
    builder.build_task_context = AsyncMock(return_value={'task': 'context'})
    return builder


@pytest.fixture
def mock_decision_engine():
    """Mock decision engine."""
    engine = Mock()
    decision = Mock()
    decision.recommended_service = 'claude_code'
    engine.analyze_task = AsyncMock(return_value=decision)
    return engine, decision


@pytest.fixture
def ai_interface(mock_config_manager):
    """Create AI interface with mocked dependencies."""
    with patch('claude_tiu.integrations.ai_interface.ClaudeCodeClient'), \
         patch('claude_tiu.integrations.ai_interface.ClaudeFlowClient'), \
         patch('claude_tiu.integrations.ai_interface.AntiHallucinationIntegration'), \
         patch('claude_tiu.integrations.ai_interface.ContextBuilder'), \
         patch('claude_tiu.integrations.ai_interface.IntegrationDecisionEngine'):
        
        interface = AIInterface(mock_config_manager)
        return interface


@pytest.fixture
def sample_project():
    """Create sample project for testing."""
    project = Mock(spec=Project)
    project.name = 'test_project'
    project.path = '/path/to/project'
    project.config = None
    return project


@pytest.fixture
def sample_task(sample_project):
    """Create sample development task."""
    task = DevelopmentTask(
        name='Test Task',
        description='Test task description',
        task_type=TaskType.CODE_GENERATION,
        priority=TaskPriority.HIGH,
        project=sample_project
    )
    return task


@pytest.fixture
def validation_result():
    """Create sample validation result."""
    result = ValidationResult(
        is_valid=True,
        authenticity_score=0.95,
        issues=[],
        confidence_score=0.92,
        validation_timestamp=datetime.now()
    )
    return result


class TestAIInterfaceInitialization:
    """Test AI Interface initialization."""
    
    def test_init_creates_components(self, mock_config_manager):
        """Test that initialization creates all required components."""
        with patch('claude_tiu.integrations.ai_interface.ClaudeCodeClient') as mock_cc, \
             patch('claude_tiu.integrations.ai_interface.ClaudeFlowClient') as mock_cf, \
             patch('claude_tiu.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah, \
             patch('claude_tiu.integrations.ai_interface.ContextBuilder') as mock_cb, \
             patch('claude_tiu.integrations.ai_interface.IntegrationDecisionEngine') as mock_de:
            
            interface = AIInterface(mock_config_manager)
            
            mock_cc.assert_called_once_with(mock_config_manager)
            mock_cf.assert_called_once_with(mock_config_manager)
            mock_ah.assert_called_once_with(mock_config_manager)
            mock_cb.assert_called_once_with(mock_config_manager)
            mock_de.assert_called_once_with(mock_config_manager)
            
            assert interface.generation_hooks == []
            assert interface.completion_hooks == []
            assert interface.error_hooks == []
            assert interface._is_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_sets_up_anti_hallucination(self, ai_interface, mock_anti_hallucination):
        """Test that initialize sets up anti-hallucination system."""
        ai_interface.anti_hallucination = mock_anti_hallucination
        
        await ai_interface.initialize()
        
        mock_anti_hallucination.initialize.assert_called_once()
        mock_anti_hallucination.register_ai_interface_hooks.assert_called_once_with(ai_interface)
        assert ai_interface._is_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_only_once(self, ai_interface, mock_anti_hallucination):
        """Test that initialize only runs once."""
        ai_interface.anti_hallucination = mock_anti_hallucination
        
        await ai_interface.initialize()
        await ai_interface.initialize()  # Second call
        
        # Should only be called once
        mock_anti_hallucination.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_handles_errors(self, ai_interface, mock_anti_hallucination):
        """Test that initialize handles errors gracefully."""
        ai_interface.anti_hallucination = mock_anti_hallucination
        mock_anti_hallucination.initialize.side_effect = Exception("Init error")
        
        with pytest.raises(Exception, match="Init error"):
            await ai_interface.initialize()
        
        assert ai_interface._is_initialized is False


class TestHookManagement:
    """Test hook management functionality."""
    
    def test_add_generation_hook(self, ai_interface):
        """Test adding generation hooks."""
        hook = Mock()
        ai_interface.add_generation_hook(hook)
        
        assert hook in ai_interface.generation_hooks
    
    def test_add_completion_hook(self, ai_interface):
        """Test adding completion hooks."""
        hook = Mock()
        ai_interface.add_completion_hook(hook)
        
        assert hook in ai_interface.completion_hooks
    
    def test_add_error_hook(self, ai_interface):
        """Test adding error hooks."""
        hook = Mock()
        ai_interface.add_error_hook(hook)
        
        assert hook in ai_interface.error_hooks


class TestClaudeCodeExecution:
    """Test Claude Code execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_success(self, ai_interface, mock_claude_code_client, 
                                               mock_context_builder, mock_anti_hallucination, 
                                               sample_project, validation_result):
        """Test successful Claude Code execution."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock responses
        code_result = CodeResult(
            success=True,
            content="generated code",
            model_used="claude-3",
            execution_time=1.5
        )
        mock_claude_code_client.execute_coding_task.return_value = code_result
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        
        # Execute
        result = await ai_interface.execute_claude_code(
            prompt="Generate a function",
            context={'file': 'test.py'},
            project=sample_project
        )
        
        # Verify calls
        mock_context_builder.build_smart_context.assert_called_once()
        mock_claude_code_client.execute_coding_task.assert_called_once()
        mock_anti_hallucination.validate_ai_generated_content.assert_called_once()
        
        # Verify result
        assert result.success is True
        assert result.content == "generated code"
        assert result.validation_passed is True
        assert result.quality_score == 0.95
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_with_auto_fixes(self, ai_interface, mock_claude_code_client, 
                                                       mock_context_builder, mock_anti_hallucination, 
                                                       sample_project):
        """Test Claude Code execution with auto-fixes applied."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock responses
        code_result = CodeResult(
            success=True,
            content="code with issues",
            model_used="claude-3"
        )
        mock_claude_code_client.execute_coding_task.return_value = code_result
        
        # Mock validation failure with fixable issues
        validation_result = ValidationResult(
            is_valid=False,
            authenticity_score=0.7,
            issues=[Mock()],
            confidence_score=0.8
        )
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        mock_anti_hallucination.auto_fix_issues.return_value = (True, "fixed code")
        
        # Execute
        result = await ai_interface.execute_claude_code(
            prompt="Generate code",
            context={},
            project=sample_project
        )
        
        # Verify auto-fix was applied
        mock_anti_hallucination.auto_fix_issues.assert_called_once()
        assert result.content == "fixed code"
        assert result.validation_passed is True
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_validation_failure(self, ai_interface, mock_claude_code_client, 
                                                          mock_context_builder, mock_anti_hallucination, 
                                                          sample_project):
        """Test Claude Code execution with validation failure."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock responses
        code_result = CodeResult(
            success=True,
            content="invalid code",
            model_used="claude-3"
        )
        mock_claude_code_client.execute_coding_task.return_value = code_result
        
        # Mock validation failure without fixes
        validation_result = ValidationResult(
            is_valid=False,
            authenticity_score=0.5,
            issues=[Mock(), Mock()],
            confidence_score=0.6
        )
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        mock_anti_hallucination.auto_fix_issues.return_value = (False, None)
        
        # Execute
        result = await ai_interface.execute_claude_code(
            prompt="Generate code",
            context={},
            project=sample_project
        )
        
        # Verify validation failure is reflected
        assert result.validation_passed is False
        assert result.quality_score == 0.5
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_with_hooks(self, ai_interface, mock_claude_code_client, 
                                                  mock_context_builder, mock_anti_hallucination, 
                                                  sample_project, validation_result):
        """Test Claude Code execution with hooks."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Add hooks
        generation_hook = AsyncMock()
        ai_interface.add_generation_hook(generation_hook)
        
        # Mock responses
        code_result = CodeResult(success=True, content="code", model_used="claude-3")
        mock_claude_code_client.execute_coding_task.return_value = code_result
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        
        # Execute
        await ai_interface.execute_claude_code(
            prompt="Generate code",
            context={},
            project=sample_project
        )
        
        # Verify hook was called
        generation_hook.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_error_handling(self, ai_interface, mock_claude_code_client, 
                                                      mock_context_builder):
        """Test Claude Code execution error handling."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface._is_initialized = True
        
        # Add error hook
        error_hook = AsyncMock()
        ai_interface.add_error_hook(error_hook)
        
        # Mock error
        mock_claude_code_client.execute_coding_task.side_effect = Exception("Execution error")
        
        # Execute and expect error
        with pytest.raises(Exception, match="Execution error"):
            await ai_interface.execute_claude_code(
                prompt="Generate code",
                context={},
                project=None
            )
        
        # Verify error hook was called
        error_hook.assert_called_once()


class TestClaudeFlowWorkflow:
    """Test Claude Flow workflow execution."""
    
    @pytest.mark.asyncio
    async def test_run_claude_flow_workflow_success(self, ai_interface, mock_claude_flow_client, sample_project):
        """Test successful Claude Flow workflow execution."""
        # Setup mocks
        ai_interface.claude_flow_client = mock_claude_flow_client
        
        # Mock workflow request and response
        workflow_request = WorkflowRequest(
            workflow_name="test_workflow",
            parameters={"param": "value"}
        )
        workflow_result = Mock()
        workflow_result.success = True
        workflow_result.output = "workflow output"
        mock_claude_flow_client.execute_workflow.return_value = workflow_result
        
        # Execute
        result = await ai_interface.run_claude_flow_workflow(
            workflow_request=workflow_request,
            project=sample_project
        )
        
        # Verify
        mock_claude_flow_client.execute_workflow.assert_called_once_with(
            workflow_request=workflow_request,
            project=sample_project
        )
        assert result == workflow_result
    
    @pytest.mark.asyncio
    async def test_run_claude_flow_workflow_error(self, ai_interface, mock_claude_flow_client, sample_project):
        """Test Claude Flow workflow execution error."""
        # Setup mocks
        ai_interface.claude_flow_client = mock_claude_flow_client
        mock_claude_flow_client.execute_workflow.side_effect = Exception("Workflow error")
        
        # Mock workflow request
        workflow_request = WorkflowRequest(
            workflow_name="test_workflow",
            parameters={}
        )
        
        # Execute and expect error
        with pytest.raises(Exception, match="Workflow error"):
            await ai_interface.run_claude_flow_workflow(
                workflow_request=workflow_request,
                project=sample_project
            )


class TestDevelopmentTaskExecution:
    """Test development task execution."""
    
    @pytest.mark.asyncio
    async def test_execute_development_task_claude_code(self, ai_interface, sample_task, sample_project, 
                                                        mock_decision_engine, mock_anti_hallucination, 
                                                        validation_result):
        """Test development task execution using Claude Code."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock task result validation
        mock_anti_hallucination.validate_task_result.return_value = validation_result
        
        # Mock internal method
        task_result = TaskResult(
            task_id=sample_task.id,
            success=True,
            generated_content="task output",
            execution_time=2.0
        )
        ai_interface._execute_task_with_claude_code = AsyncMock(return_value=task_result)
        
        # Execute
        result = await ai_interface.execute_development_task(sample_task, sample_project)
        
        # Verify
        ai_interface.decision_engine.analyze_task.assert_called_once_with(sample_task, sample_project)
        ai_interface._execute_task_with_claude_code.assert_called_once_with(sample_task, sample_project)
        mock_anti_hallucination.validate_task_result.assert_called_once()
        
        assert result.success is True
        assert result.validation_passed is True
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_development_task_claude_flow(self, ai_interface, sample_task, sample_project, 
                                                        mock_decision_engine, mock_anti_hallucination, 
                                                        validation_result):
        """Test development task execution using Claude Flow."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        decision.recommended_service = 'claude_flow'
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock task result validation
        mock_anti_hallucination.validate_task_result.return_value = validation_result
        
        # Mock internal method
        task_result = TaskResult(
            task_id=sample_task.id,
            success=True,
            generated_content="workflow output"
        )
        ai_interface._execute_task_with_claude_flow = AsyncMock(return_value=task_result)
        
        # Execute
        result = await ai_interface.execute_development_task(sample_task, sample_project)
        
        # Verify
        ai_interface._execute_task_with_claude_flow.assert_called_once_with(sample_task, sample_project)
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_development_task_hybrid(self, ai_interface, sample_task, sample_project, 
                                                   mock_decision_engine, mock_anti_hallucination, 
                                                   validation_result):
        """Test development task execution using hybrid approach."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        decision.recommended_service = 'hybrid'
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock task result validation
        mock_anti_hallucination.validate_task_result.return_value = validation_result
        
        # Mock internal method
        task_result = TaskResult(
            task_id=sample_task.id,
            success=True,
            generated_content="hybrid output"
        )
        ai_interface._execute_hybrid_task = AsyncMock(return_value=task_result)
        
        # Execute
        result = await ai_interface.execute_development_task(sample_task, sample_project)
        
        # Verify
        ai_interface._execute_hybrid_task.assert_called_once_with(sample_task, sample_project, decision)
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_development_task_with_completion_hooks(self, ai_interface, sample_task, 
                                                                 sample_project, mock_decision_engine, 
                                                                 mock_anti_hallucination, validation_result):
        """Test development task execution with completion hooks."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Add completion hook
        completion_hook = AsyncMock()
        ai_interface.add_completion_hook(completion_hook)
        
        # Mock task result validation
        mock_anti_hallucination.validate_task_result.return_value = validation_result
        
        # Mock internal method
        task_result = TaskResult(task_id=sample_task.id, success=True)
        ai_interface._execute_task_with_claude_code = AsyncMock(return_value=task_result)
        
        # Execute
        await ai_interface.execute_development_task(sample_task, sample_project)
        
        # Verify completion hook was called
        completion_hook.assert_called_once_with(sample_task, task_result, sample_project)
    
    @pytest.mark.asyncio
    async def test_execute_development_task_error_handling(self, ai_interface, sample_task, sample_project, 
                                                          mock_decision_engine):
        """Test development task execution error handling."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        ai_interface._is_initialized = True
        
        # Add error hook
        error_hook = AsyncMock()
        ai_interface.add_error_hook(error_hook)
        
        # Mock internal method to raise error
        ai_interface._execute_task_with_claude_code = AsyncMock(side_effect=Exception("Task error"))
        
        # Execute
        result = await ai_interface.execute_development_task(sample_task, sample_project)
        
        # Verify error handling
        assert result.success is False
        assert "Task error" in result.error_message
        assert result.validation_passed is False
        
        # Verify error hook was called
        error_hook.assert_called_once()


class TestValidationMethods:
    """Test AI output validation methods."""
    
    @pytest.mark.asyncio
    async def test_validate_ai_output(self, ai_interface, sample_task, sample_project, 
                                      mock_anti_hallucination, validation_result):
        """Test AI output validation."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        
        # Execute
        result = await ai_interface.validate_ai_output(
            output="generated code",
            task=sample_task,
            project=sample_project
        )
        
        # Verify
        mock_anti_hallucination.validate_ai_generated_content.assert_called_once()
        assert result == validation_result
    
    @pytest.mark.asyncio
    async def test_get_validation_metrics(self, ai_interface, mock_anti_hallucination):
        """Test getting validation metrics."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        metrics = {'accuracy': 0.95, 'precision': 0.92}
        mock_anti_hallucination.get_integration_metrics.return_value = metrics
        
        # Execute
        result = await ai_interface.get_validation_metrics()
        
        # Verify
        mock_anti_hallucination.get_integration_metrics.assert_called_once()
        assert result == metrics
    
    @pytest.mark.asyncio
    async def test_validate_project_codebase(self, ai_interface, sample_project, mock_anti_hallucination):
        """Test project codebase validation."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        validation_results = {
            'file1.py': ValidationResult(is_valid=True, authenticity_score=0.95),
            'file2.py': ValidationResult(is_valid=False, authenticity_score=0.7)
        }
        mock_anti_hallucination.validate_project_codebase.return_value = validation_results
        
        # Execute
        result = await ai_interface.validate_project_codebase(sample_project)
        
        # Verify
        mock_anti_hallucination.validate_project_codebase.assert_called_once_with(
            project=sample_project,
            incremental=True
        )
        assert result == validation_results
    
    @pytest.mark.asyncio
    async def test_retrain_anti_hallucination_models(self, ai_interface, mock_anti_hallucination):
        """Test retraining anti-hallucination models."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        training_results = {'model_accuracy': 0.97, 'training_time': 120}
        mock_anti_hallucination.retrain_models.return_value = training_results
        
        # Execute
        result = await ai_interface.retrain_anti_hallucination_models(
            project_samples=True,
            full_retrain=False
        )
        
        # Verify
        mock_anti_hallucination.retrain_models.assert_called_once_with(
            additional_samples=None,
            full_retrain=False
        )
        assert result == training_results


class TestStreamingValidation:
    """Test streaming validation functionality."""
    
    @pytest.mark.asyncio
    async def test_validate_code_streaming(self, ai_interface, mock_anti_hallucination):
        """Test streaming code validation."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock validation for each chunk
        chunk_validation = ValidationResult(
            is_valid=True,
            authenticity_score=0.9,
            issues=[]
        )
        mock_anti_hallucination.validate_ai_generated_content.return_value = chunk_validation
        
        # Test with code longer than chunk size
        code_stream = "def function():\n    pass\n" * 50  # Long code
        context = {'mode': 'streaming'}
        
        # Execute
        result = await ai_interface.validate_code_streaming(code_stream, context)
        
        # Verify
        assert result['streaming_validation'] is True
        assert result['chunks_validated'] > 1
        assert result['avg_authenticity_score'] == 0.9
        assert result['total_issues'] == 0
        assert 'validation_chunks' in result
    
    @pytest.mark.asyncio
    async def test_get_real_time_metrics(self, ai_interface, mock_anti_hallucination):
        """Test getting real-time metrics."""
        # Setup mocks
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Add some hooks and requests
        ai_interface.add_generation_hook(Mock())
        ai_interface.add_completion_hook(Mock())
        ai_interface._active_requests['req1'] = Mock()
        ai_interface._request_history = [Mock(), Mock()]
        
        # Mock base metrics
        base_metrics = {
            'integration_metrics': {
                'avg_validation_time': 0.5,
                'cache_hit_rate': 0.8,
                'success_rate': 0.95
            }
        }
        mock_anti_hallucination.get_integration_metrics.return_value = base_metrics
        ai_interface.get_validation_metrics = AsyncMock(return_value=base_metrics)
        
        # Execute
        result = await ai_interface.get_real_time_metrics()
        
        # Verify
        assert 'real_time_validations' in result
        assert result['real_time_validations']['hooks_registered'] == 2
        assert result['real_time_validations']['active_requests'] == 1
        assert result['real_time_validations']['request_history_size'] == 2
        assert 'performance_metrics' in result


class TestCleanup:
    """Test cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_cleanup(self, ai_interface, mock_claude_code_client, mock_claude_flow_client, 
                          mock_anti_hallucination):
        """Test AI interface cleanup."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.claude_flow_client = mock_claude_flow_client
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Add some active requests
        ai_interface._active_requests['req1'] = Mock()
        ai_interface._active_requests['req2'] = Mock()
        
        # Execute cleanup
        await ai_interface.cleanup()
        
        # Verify cleanup calls
        mock_anti_hallucination.cleanup.assert_called_once()
        mock_claude_code_client.cleanup.assert_called_once()
        mock_claude_flow_client.cleanup.assert_called_once()
        
        # Verify state reset
        assert len(ai_interface._active_requests) == 0
        assert ai_interface._is_initialized is False


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_not_initialized(self, ai_interface):
        """Test Claude Code execution when not initialized."""
        ai_interface._is_initialized = False
        ai_interface.initialize = AsyncMock()
        ai_interface.claude_code_client = Mock()
        ai_interface.context_builder = Mock()
        ai_interface.anti_hallucination = Mock()
        
        # Mock the rest of the execution chain
        code_result = CodeResult(success=True, content="code")
        ai_interface.claude_code_client.execute_coding_task = AsyncMock(return_value=code_result)
        ai_interface.context_builder.build_smart_context = AsyncMock(return_value={})
        ai_interface.anti_hallucination.validate_ai_generated_content = AsyncMock(
            return_value=ValidationResult(is_valid=True, authenticity_score=0.9)
        )
        
        # Execute
        await ai_interface.execute_claude_code("prompt", {})
        
        # Verify initialization was called
        ai_interface.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hook_error_handling(self, ai_interface, mock_claude_code_client, 
                                       mock_context_builder, mock_anti_hallucination, validation_result):
        """Test that hook errors don't break execution."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Add failing hook
        failing_hook = AsyncMock(side_effect=Exception("Hook error"))
        ai_interface.add_generation_hook(failing_hook)
        
        # Mock responses
        code_result = CodeResult(success=True, content="code", model_used="claude-3")
        mock_claude_code_client.execute_coding_task.return_value = code_result
        mock_anti_hallucination.validate_ai_generated_content.return_value = validation_result
        
        # Execute - should not raise exception despite failing hook
        result = await ai_interface.execute_claude_code("prompt", {})
        
        # Verify execution continued
        assert result.success is True
        assert result.content == "code"
    
    @pytest.mark.asyncio
    async def test_correct_task_result(self, ai_interface, sample_task, mock_claude_code_client):
        """Test task result correction."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        
        # Mock original result with issues
        original_result = TaskResult(
            task_id=sample_task.id,
            success=False,
            generated_content="buggy code",
            execution_time=1.0
        )
        
        # Mock correction result
        correction_result = CodeResult(
            success=True,
            content="fixed code",
            model_used="claude-3",
            execution_time=0.5
        )
        mock_claude_code_client.execute_coding_task.return_value = correction_result
        
        # Execute correction
        result = await ai_interface.correct_task_result(
            task=sample_task,
            result=original_result,
            validation_issues=["Issue 1", "Issue 2"]
        )
        
        # Verify
        assert result.success is True
        assert result.generated_content == "fixed code"
        assert result.execution_time == 1.5  # Original + correction time
        assert result.attempts == 1  # Original attempts + 1
    
    @pytest.mark.asyncio
    async def test_analyze_requirements(self, ai_interface, sample_project, mock_claude_code_client, 
                                        mock_context_builder):
        """Test requirements analysis."""
        # Setup mocks
        ai_interface.claude_code_client = mock_claude_code_client
        ai_interface.context_builder = mock_context_builder
        
        # Mock responses
        analysis_result = CodeResult(
            success=True,
            content="analysis output",
            model_used="claude-3"
        )
        mock_claude_code_client.execute_coding_task.return_value = analysis_result
        mock_context_builder.build_project_context.return_value = {'project': 'context'}
        
        # Mock internal parsing method
        ai_interface._parse_requirements_analysis = AsyncMock(return_value=Mock())
        
        # Execute
        requirements = {'feature': 'description'}
        result = await ai_interface.analyze_requirements(requirements, sample_project)
        
        # Verify
        mock_context_builder.build_project_context.assert_called_once_with(sample_project)
        mock_claude_code_client.execute_coding_task.assert_called_once()
        ai_interface._parse_requirements_analysis.assert_called_once_with(analysis_result, sample_project)
        assert result is not None


@pytest.mark.integration
class TestAIInterfaceIntegration:
    """Integration tests for AI Interface."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, mock_config_manager, sample_project, sample_task):
        """Test full workflow integration from task to result."""
        with patch('claude_tiu.integrations.ai_interface.ClaudeCodeClient') as mock_cc_class, \
             patch('claude_tiu.integrations.ai_interface.ClaudeFlowClient') as mock_cf_class, \
             patch('claude_tiu.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class, \
             patch('claude_tiu.integrations.ai_interface.ContextBuilder') as mock_cb_class, \
             patch('claude_tiu.integrations.ai_interface.IntegrationDecisionEngine') as mock_de_class:
            
            # Setup mock instances
            mock_cc = Mock()
            mock_cf = Mock()
            mock_ah = Mock()
            mock_cb = Mock()
            mock_de = Mock()
            
            mock_cc_class.return_value = mock_cc
            mock_cf_class.return_value = mock_cf
            mock_ah_class.return_value = mock_ah
            mock_cb_class.return_value = mock_cb
            mock_de_class.return_value = mock_de
            
            # Setup async methods
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_cc.cleanup = AsyncMock()
            mock_cf.cleanup = AsyncMock()
            mock_ah.cleanup = AsyncMock()
            
            # Setup decision engine
            decision = Mock()
            decision.recommended_service = 'claude_code'
            mock_de.analyze_task = AsyncMock(return_value=decision)
            
            # Setup context builder
            mock_cb.build_task_context = AsyncMock(return_value={'context': 'built'})
            
            # Setup Claude Code execution
            code_result = CodeResult(
                success=True,
                content="def hello():\n    return 'world'",
                model_used="claude-3",
                execution_time=1.5
            )
            mock_cc.execute_coding_task = AsyncMock(return_value=code_result)
            
            # Setup validation
            validation_result = ValidationResult(
                is_valid=True,
                authenticity_score=0.95,
                issues=[],
                confidence_score=0.92
            )
            mock_ah.validate_task_result = AsyncMock(return_value=validation_result)
            
            # Create and initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Execute full workflow
            result = await ai_interface.execute_development_task(sample_task, sample_project)
            
            # Verify full workflow
            assert result.success is True
            assert result.validation_passed is True
            assert result.generated_content == "def hello():\n    return 'world'"
            assert result.execution_time == 1.5
            
            # Verify all components were used
            mock_de.analyze_task.assert_called_once()
            mock_cb.build_task_context.assert_called_once()
            mock_cc.execute_coding_task.assert_called_once()
            mock_ah.validate_task_result.assert_called_once()


@pytest.mark.performance
class TestAIInterfacePerformance:
    """Performance tests for AI Interface."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, ai_interface, sample_project, mock_decision_engine, 
                                            mock_anti_hallucination, validation_result):
        """Test concurrent task execution performance."""
        # Setup mocks
        ai_interface.decision_engine, decision = mock_decision_engine
        ai_interface.anti_hallucination = mock_anti_hallucination
        ai_interface._is_initialized = True
        
        # Mock validation
        mock_anti_hallucination.validate_task_result.return_value = validation_result
        
        # Create multiple tasks
        tasks = []
        for i in range(10):
            task = DevelopmentTask(
                name=f'Task {i}',
                description=f'Test task {i}',
                task_type=TaskType.CODE_GENERATION,
                priority=TaskPriority.MEDIUM,
                project=sample_project
            )
            tasks.append(task)
        
        # Mock internal execution method
        task_result = TaskResult(
            task_id="test",
            success=True,
            generated_content="output",
            execution_time=0.1
        )
        ai_interface._execute_task_with_claude_code = AsyncMock(return_value=task_result)
        
        # Execute tasks concurrently
        import time
        start_time = time.time()
        
        results = await asyncio.gather(*[
            ai_interface.execute_development_task(task, sample_project)
            for task in tasks
        ])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all tasks completed successfully
        assert len(results) == 10
        assert all(result.success for result in results)
        
        # Should be significantly faster than sequential execution
        # (This is a rough check - adjust threshold based on your requirements)
        assert total_time < 2.0  # Should complete within 2 seconds
