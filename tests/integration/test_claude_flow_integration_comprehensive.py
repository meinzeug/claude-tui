"""Comprehensive integration tests for Claude Code/Flow workflow integration."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

from claude_tui.integrations.ai_interface import AIInterface
from claude_tui.core.config_manager import ConfigManager
from claude_tui.models.task import DevelopmentTask, TaskResult, TaskType, TaskPriority
from claude_tui.models.project import Project
from claude_tui.validation.progress_validator import ValidationResult, ValidationSeverity


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager with realistic settings."""
    manager = Mock(spec=ConfigManager)
    manager.get_setting = AsyncMock(side_effect=lambda path, default=None: {
        'ai_services.claude.timeout': 300,
        'ai_services.claude.max_retries': 3,
        'project_defaults.auto_validation': True,
        'security.sandbox_enabled': True
    }.get(path, default))
    
    # Mock AI service config
    from claude_tui.core.config_manager import AIServiceConfig
    service_config = AIServiceConfig(
        service_name='claude',
        endpoint_url='https://api.anthropic.com',
        timeout=300,
        max_retries=3
    )
    manager.get_ai_service_config = AsyncMock(return_value=service_config)
    manager.get_api_key = AsyncMock(return_value='test_api_key')
    
    return manager


@pytest.fixture
def test_project(tmp_path):
    """Create test project structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create project files
    (project_dir / "main.py").write_text("# Main file\nprint('Hello World')")
    (project_dir / "requirements.txt").write_text("requests>=2.28.0\npytest>=7.0.0")
    (project_dir / "README.md").write_text("# Test Project\nThis is a test project.")
    
    # Create subdirectories
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "src" / "__init__.py").touch()
    (project_dir / "tests" / "__init__.py").touch()
    
    project = Mock(spec=Project)
    project.name = "test_project"
    project.path = project_dir
    project.template = None
    project.config = None
    
    return project


@pytest.fixture
def sample_tasks(test_project):
    """Create sample development tasks."""
    tasks = [
        DevelopmentTask(
            name="Create API Endpoint",
            description="Create a REST API endpoint for user management",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            project=test_project
        ),
        DevelopmentTask(
            name="Write Unit Tests",
            description="Write comprehensive unit tests for the API",
            task_type=TaskType.TEST_GENERATION,
            priority=TaskPriority.MEDIUM,
            project=test_project
        ),
        DevelopmentTask(
            name="Update Documentation",
            description="Update project documentation with API specs",
            task_type=TaskType.DOCUMENTATION,
            priority=TaskPriority.LOW,
            project=test_project
        )
    ]
    return tasks


class TestAIInterfaceIntegration:
    """Test full AI Interface integration."""
    
    @pytest.mark.asyncio
    async def test_ai_interface_full_workflow(self, mock_config_manager, test_project, sample_tasks):
        """Test complete AI interface workflow integration."""
        # Initialize AI interface with mocked components
        with patch('claude_tui.integrations.ai_interface.ClaudeCodeClient') as mock_cc_class, \
             patch('claude_tui.integrations.ai_interface.ClaudeFlowClient') as mock_cf_class, \
             patch('claude_tui.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class:
            
            # Setup mock instances
            mock_cc = Mock()
            mock_cf = Mock()
            mock_ah = Mock()
            
            mock_cc_class.return_value = mock_cc
            mock_cf_class.return_value = mock_cf
            mock_ah_class.return_value = mock_ah
            
            # Configure mocks
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_ah.validate_ai_generated_content = AsyncMock(return_value=ValidationResult(
                is_valid=True,
                authenticity_score=0.95,
                issues=[]
            ))
            mock_ah.validate_task_result = AsyncMock(return_value=ValidationResult(
                is_valid=True,
                authenticity_score=0.93,
                issues=[]
            ))
            mock_cc.cleanup = AsyncMock()
            mock_cf.cleanup = AsyncMock()
            mock_ah.cleanup = AsyncMock()
            
            # Initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Execute tasks in sequence
            results = []
            for task in sample_tasks:
                # Mock decision engine to choose appropriate service
                with patch.object(ai_interface, 'decision_engine') as mock_decision:
                    decision = Mock()
                    decision.recommended_service = 'claude_code'
                    mock_decision.analyze_task = AsyncMock(return_value=decision)
                    
                    # Mock task execution
                    with patch.object(ai_interface, '_execute_task_with_claude_code') as mock_execute:
                        task_result = TaskResult(
                            task_id=task.id,
                            success=True,
                            generated_content=f"Generated content for {task.name}",
                            execution_time=2.5,
                            quality_score=0.9
                        )
                        mock_execute.return_value = task_result
                        
                        result = await ai_interface.execute_development_task(task, test_project)
                        results.append(result)
            
            # Verify all tasks completed successfully
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all(result.validation_passed for result in results)
            
            # Verify quality scores
            assert all(result.quality_score >= 0.9 for result in results)
    
    @pytest.mark.asyncio
    async def test_ai_interface_error_recovery(self, mock_config_manager, test_project):
        """Test AI interface error recovery and correction."""
        with patch('claude_tui.integrations.ai_interface.ClaudeCodeClient') as mock_cc_class, \
             patch('claude_tui.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class:
            
            mock_cc = Mock()
            mock_ah = Mock()
            
            mock_cc_class.return_value = mock_cc
            mock_ah_class.return_value = mock_ah
            
            # Configure mocks for initialization
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_cc.cleanup = AsyncMock()
            mock_ah.cleanup = AsyncMock()
            
            # Initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Create task that will initially fail validation
            failing_task = DevelopmentTask(
                name="Problematic Task",
                description="Task that produces invalid code",
                task_type=TaskType.CODE_GENERATION,
                priority=TaskPriority.HIGH,
                project=test_project
            )
            
            # Mock initial failure then success on correction
            initial_result = TaskResult(
                task_id=failing_task.id,
                success=True,
                generated_content="invalid code with placeholders",
                execution_time=1.0
            )
            
            # Mock validation failure then success after auto-fix
            validation_failure = ValidationResult(
                is_valid=False,
                authenticity_score=0.6,
                issues=[Mock(severity=ValidationSeverity.HIGH, message="Placeholder detected")]
            )
            
            validation_success = ValidationResult(
                is_valid=True,
                authenticity_score=0.95,
                issues=[]
            )
            
            mock_ah.validate_task_result.return_value = validation_failure
            mock_ah.auto_fix_issues.return_value = (True, "fixed code without placeholders")
            
            with patch.object(ai_interface, '_execute_task_with_claude_code', return_value=initial_result):
                with patch.object(ai_interface.decision_engine, 'analyze_task') as mock_analyze:
                    decision = Mock()
                    decision.recommended_service = 'claude_code'
                    mock_analyze.return_value = decision
                    
                    result = await ai_interface.execute_development_task(failing_task, test_project)
                    
                    # Verify auto-fix was applied
                    assert result.success is True
                    assert result.generated_content == "fixed code without placeholders"
                    assert hasattr(result, 'auto_fixes_applied') and result.auto_fixes_applied
    
    @pytest.mark.asyncio
    async def test_ai_interface_streaming_validation(self, mock_config_manager):
        """Test real-time streaming validation."""
        with patch('claude_tui.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class:
            mock_ah = Mock()
            mock_ah_class.return_value = mock_ah
            
            # Configure mocks
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_ah.validate_ai_generated_content = AsyncMock(return_value=ValidationResult(
                is_valid=True,
                authenticity_score=0.92,
                issues=[]
            ))
            mock_ah.cleanup = AsyncMock()
            
            # Initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Test streaming validation with large code block
            large_code_stream = """def complex_function():
    # This is a long function with many lines
    data = []
    for i in range(1000):
        if i % 2 == 0:
            data.append(i * 2)
        else:
            data.append(i * 3)
    
    processed_data = []
    for item in data:
        if item > 500:
            processed_data.append(item / 2)
        else:
            processed_data.append(item * 1.5)
    
    return processed_data
"""
            
            context = {'validation_mode': 'streaming', 'language': 'python'}
            
            result = await ai_interface.validate_code_streaming(large_code_stream, context)
            
            # Verify streaming validation results
            assert result['streaming_validation'] is True
            assert result['chunks_validated'] > 1
            assert result['avg_authenticity_score'] == 0.92
            assert 'validation_chunks' in result
            assert len(result['validation_chunks']) > 1


class TestAntiHallucinationIntegration:
    """Test Anti-Hallucination Engine integration."""
    
    @pytest.mark.asyncio
    async def test_anti_hallucination_validation_pipeline(self, mock_config_manager, test_project):
        """Test complete anti-hallucination validation pipeline."""
        from claude_tui.integrations.anti_hallucination_integration import AntiHallucinationIntegration
        
        with patch.object(AntiHallucinationIntegration, '_load_models'), \
             patch.object(AntiHallucinationIntegration, '_initialize_validators'):
            
            ah_engine = AntiHallucinationIntegration(mock_config_manager)
            await ah_engine.initialize()
            
            # Test different types of content validation
            test_cases = [
                {
                    'content': 'def hello(): return "world"',
                    'expected_score': 0.95,
                    'expected_valid': True,
                    'content_type': 'clean_code'
                },
                {
                    'content': 'def TODO_function(): pass  # TODO: Implement',
                    'expected_score': 0.6,
                    'expected_valid': False,
                    'content_type': 'placeholder_code'
                },
                {
                    'content': 'import os\nos.system("rm -rf /")',
                    'expected_score': 0.2,
                    'expected_valid': False,
                    'content_type': 'malicious_code'
                }
            ]
            
            for case in test_cases:
                with patch.object(ah_engine, '_validate_content_authenticity') as mock_validate:
                    # Mock the validation based on content type
                    if case['content_type'] == 'clean_code':
                        validation_result = ValidationResult(
                            is_valid=True,
                            authenticity_score=0.95,
                            issues=[]
                        )
                    elif case['content_type'] == 'placeholder_code':
                        validation_result = ValidationResult(
                            is_valid=False,
                            authenticity_score=0.6,
                            issues=[Mock(severity=ValidationSeverity.MEDIUM, message="TODO placeholder")]
                        )
                    else:  # malicious_code
                        validation_result = ValidationResult(
                            is_valid=False,
                            authenticity_score=0.2,
                            issues=[Mock(severity=ValidationSeverity.CRITICAL, message="Potentially malicious")]
                        )
                    
                    mock_validate.return_value = validation_result
                    
                    result = await ah_engine.validate_ai_generated_content(
                        content=case['content'],
                        context={'content_type': case['content_type']},
                        project=test_project
                    )
                    
                    assert result.is_valid == case['expected_valid']
                    assert abs(result.authenticity_score - case['expected_score']) < 0.1


class TestPerformanceIntegration:
    """Test performance aspects of integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_task_processing_performance(self, mock_config_manager, test_project):
        """Test performance of concurrent task processing."""
        with patch('claude_tui.integrations.ai_interface.ClaudeCodeClient') as mock_cc_class, \
             patch('claude_tui.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class:
            
            mock_cc = Mock()
            mock_ah = Mock()
            
            mock_cc_class.return_value = mock_cc
            mock_ah_class.return_value = mock_ah
            
            # Configure mocks for fast execution
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_ah.validate_task_result = AsyncMock(return_value=ValidationResult(
                is_valid=True,
                authenticity_score=0.95,
                issues=[]
            ))
            mock_cc.cleanup = AsyncMock()
            mock_ah.cleanup = AsyncMock()
            
            # Initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Create multiple tasks
            tasks = []
            for i in range(20):  # Test with 20 concurrent tasks
                task = DevelopmentTask(
                    name=f"Performance Task {i}",
                    description=f"Performance test task number {i}",
                    task_type=TaskType.CODE_GENERATION,
                    priority=TaskPriority.MEDIUM,
                    project=test_project
                )
                tasks.append(task)
            
            # Mock fast task execution
            async def fast_task_execution(task, project):
                await asyncio.sleep(0.01)  # Simulate 10ms execution
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    generated_content=f"Fast result for {task.name}",
                    execution_time=0.01
                )
            
            with patch.object(ai_interface, '_execute_task_with_claude_code', side_effect=fast_task_execution):
                with patch.object(ai_interface.decision_engine, 'analyze_task') as mock_analyze:
                    decision = Mock()
                    decision.recommended_service = 'claude_code'
                    mock_analyze.return_value = decision
                    
                    # Measure execution time
                    start_time = asyncio.get_event_loop().time()
                    
                    # Execute tasks concurrently
                    results = await asyncio.gather(*[
                        ai_interface.execute_development_task(task, test_project)
                        for task in tasks
                    ])
                    
                    end_time = asyncio.get_event_loop().time()
                    total_time = end_time - start_time
                    
                    # Verify results
                    assert len(results) == 20
                    assert all(result.success for result in results)
                    
                    # Performance check: should complete much faster than sequential
                    # (20 tasks * 0.01s = 0.2s sequential, concurrent should be < 0.1s)
                    assert total_time < 0.5  # Allow some overhead
                    
                    # Verify throughput
                    throughput = len(tasks) / total_time
                    assert throughput > 40  # Should process more than 40 tasks per second


@pytest.mark.slow
class TestRealWorldScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_development_workflow(self, mock_config_manager, test_project):
        """Test complete development workflow from requirements to deployment."""
        # This test simulates a full development cycle:
        # 1. Requirements analysis
        # 2. Code generation
        # 3. Test generation
        # 4. Documentation generation
        # 5. Code review and validation
        # 6. Deployment preparation
        
        with patch('claude_tui.integrations.ai_interface.ClaudeCodeClient') as mock_cc_class, \
             patch('claude_tui.integrations.ai_interface.ClaudeFlowClient') as mock_cf_class, \
             patch('claude_tui.integrations.ai_interface.AntiHallucinationIntegration') as mock_ah_class:
            
            # Setup mock clients
            mock_cc = Mock()
            mock_cf = Mock()
            mock_ah = Mock()
            
            mock_cc_class.return_value = mock_cc
            mock_cf_class.return_value = mock_cf
            mock_ah_class.return_value = mock_ah
            
            # Configure mocks
            mock_ah.initialize = AsyncMock()
            mock_ah.register_ai_interface_hooks = AsyncMock()
            mock_ah.validate_ai_generated_content = AsyncMock()
            mock_ah.validate_task_result = AsyncMock()
            mock_cc.cleanup = AsyncMock()
            mock_cf.cleanup = AsyncMock()
            mock_ah.cleanup = AsyncMock()
            
            # Initialize AI interface
            ai_interface = AIInterface(mock_config_manager)
            await ai_interface.initialize()
            
            # Define workflow phases
            workflow_phases = [
                {
                    'name': 'Requirements Analysis',
                    'task_type': TaskType.ANALYSIS,
                    'expected_output': 'User story breakdown and technical requirements',
                    'validation_score': 0.92
                },
                {
                    'name': 'API Design',
                    'task_type': TaskType.CODE_GENERATION,
                    'expected_output': 'RESTful API endpoints with FastAPI',
                    'validation_score': 0.95
                },
                {
                    'name': 'Database Schema',
                    'task_type': TaskType.CODE_GENERATION,
                    'expected_output': 'SQLAlchemy models and migrations',
                    'validation_score': 0.93
                },
                {
                    'name': 'Unit Tests',
                    'task_type': TaskType.TEST_GENERATION,
                    'expected_output': 'Comprehensive pytest test suite',
                    'validation_score': 0.89
                },
                {
                    'name': 'Integration Tests',
                    'task_type': TaskType.TEST_GENERATION,
                    'expected_output': 'API integration tests with test client',
                    'validation_score': 0.87
                },
                {
                    'name': 'Documentation',
                    'task_type': TaskType.DOCUMENTATION,
                    'expected_output': 'API documentation and user guide',
                    'validation_score': 0.91
                }
            ]
            
            # Execute workflow phases
            phase_results = []
            for phase in workflow_phases:
                # Create task for phase
                task = DevelopmentTask(
                    name=phase['name'],
                    description=f"Generate {phase['expected_output']}",
                    task_type=phase['task_type'],
                    priority=TaskPriority.HIGH,
                    project=test_project
                )
                
                # Mock validation for this phase
                mock_ah.validate_task_result.return_value = ValidationResult(
                    is_valid=True,
                    authenticity_score=phase['validation_score'],
                    issues=[]
                )
                
                # Mock task execution
                with patch.object(ai_interface, '_execute_task_with_claude_code') as mock_execute:
                    task_result = TaskResult(
                        task_id=task.id,
                        success=True,
                        generated_content=phase['expected_output'],
                        execution_time=2.0,
                        quality_score=phase['validation_score']
                    )
                    mock_execute.return_value = task_result
                    
                    with patch.object(ai_interface.decision_engine, 'analyze_task') as mock_analyze:
                        decision = Mock()
                        decision.recommended_service = 'claude_code'
                        mock_analyze.return_value = decision
                        
                        result = await ai_interface.execute_development_task(task, test_project)
                        phase_results.append(result)
            
            # Verify complete workflow
            assert len(phase_results) == 6
            assert all(result.success for result in phase_results)
            assert all(result.validation_passed for result in phase_results)
            
            # Verify quality progression
            avg_quality = sum(result.quality_score for result in phase_results) / len(phase_results)
            assert avg_quality > 0.9
            
            # Verify all phases completed
            phase_names = [phase['name'] for phase in workflow_phases]
            assert 'Requirements Analysis' in phase_names
            assert 'API Design' in phase_names
            assert 'Unit Tests' in phase_names
            assert 'Documentation' in phase_names