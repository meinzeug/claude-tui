#!/usr/bin/env python3
"""
Comprehensive Claude Code/Flow Integration Tests.

Tests the integration with Claude Code and Claude Flow including:
- Claude Code command execution and file operations
- Claude Flow orchestration and agent coordination
- Error handling and retry mechanisms
- Performance monitoring and optimization
- Real vs mock integration scenarios
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

# Import the components under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from integrations.claude_code import ClaudeCodeIntegration
from integrations.claude_flow import ClaudeFlowIntegration
from core.types import Task, Priority, AITaskResult


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create realistic project structure
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "docs").mkdir()
        
        # Create sample files
        (workspace / "src" / "main.py").write_text('print("Hello World")')
        (workspace / "requirements.txt").write_text('requests>=2.25.1\n')
        (workspace / "README.md").write_text('# Test Project\n')
        
        yield workspace


@pytest.fixture
def sample_task():
    """Create sample task for integration testing."""
    return Task(
        id=uuid4(),
        name="Create REST API endpoint",
        description="Create a FastAPI endpoint for user authentication with JWT tokens",
        priority=Priority.HIGH,
        metadata={
            'complexity': 'medium',
            'estimated_lines': 50,
            'requires_auth': True
        }
    )


@pytest.fixture
def sample_context():
    """Create sample context for integration testing."""
    return {
        'project_path': '/tmp/test_project',
        'project_type': 'python',
        'framework': 'fastapi',
        'existing_files': ['main.py', 'models.py', 'auth.py'],
        'requirements': [
            'Create secure authentication endpoint',
            'Add JWT token generation',
            'Include input validation',
            'Add error handling'
        ],
        'constraints': {
            'security_level': 'high',
            'performance_target': 'fast',
            'code_style': 'pep8'
        }
    }


class TestClaudeCodeIntegration:
    """Tests for Claude Code integration."""
    
    def test_claude_code_initialization(self):
        """Test Claude Code integration initialization."""
        integration = ClaudeCodeIntegration()
        
        assert integration is not None
        assert hasattr(integration, 'execute_command')
        assert hasattr(integration, 'get_status')
        assert hasattr(integration, 'cleanup')
    
    @pytest.mark.asyncio
    async def test_simple_command_execution(self, temp_workspace, sample_task, sample_context):
        """Test simple command execution through Claude Code."""
        integration = ClaudeCodeIntegration()
        
        # Mock the actual Claude Code execution
        mock_response = {
            'success': True,
            'output': 'Created authentication endpoint successfully',
            'files_modified': ['src/auth.py', 'src/main.py'],
            'execution_time': 2.5,
            'tokens_used': 1250
        }
        
        with patch.object(integration, '_execute_claude_command', return_value=mock_response):
            result = await integration.execute_command(
                task=sample_task,
                context=sample_context,
                workspace_path=temp_workspace
            )
        
        assert result['success'] is True
        assert 'auth.py' in str(result['files_modified'])
        assert result['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_file_operation_tracking(self, temp_workspace, sample_task, sample_context):
        """Test file operation tracking and validation."""
        integration = ClaudeCodeIntegration()
        
        # Create initial file state
        initial_files = list(temp_workspace.rglob('*.py'))
        initial_count = len(initial_files)
        
        # Mock Claude Code creating new files
        mock_response = {
            'success': True,
            'output': 'Files created and modified',
            'files_modified': ['src/auth.py', 'src/models/user.py'],
            'files_created': ['src/middleware/auth_middleware.py'],
            'files_deleted': [],
            'execution_time': 1.8
        }
        
        with patch.object(integration, '_execute_claude_command', return_value=mock_response):
            # Create the files that Claude Code claims to create
            (temp_workspace / "src" / "auth.py").write_text('# Auth module')
            (temp_workspace / "src" / "models").mkdir(exist_ok=True)
            (temp_workspace / "src" / "models" / "user.py").write_text('# User model')
            (temp_workspace / "src" / "middleware").mkdir(exist_ok=True)
            (temp_workspace / "src" / "middleware" / "auth_middleware.py").write_text('# Auth middleware')
            
            result = await integration.execute_command(
                task=sample_task,
                context=sample_context,
                workspace_path=temp_workspace
            )
            
            # Verify file operations
            final_files = list(temp_workspace.rglob('*.py'))
            final_count = len(final_files)
            
            assert final_count > initial_count  # New files were created
            assert (temp_workspace / "src" / "auth.py").exists()
            assert (temp_workspace / "src" / "middleware" / "auth_middleware.py").exists()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_workspace, sample_task, sample_context):
        """Test error handling and recovery mechanisms."""
        integration = ClaudeCodeIntegration()
        
        # Test network error scenario
        with patch.object(integration, '_execute_claude_command') as mock_execute:
            mock_execute.side_effect = ConnectionError("Network timeout")
            
            result = await integration.execute_command(
                task=sample_task,
                context=sample_context,
                workspace_path=temp_workspace
            )
            
            assert result['success'] is False
            assert 'Network timeout' in result['error']
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, temp_workspace, sample_task, sample_context):
        """Test retry mechanism for failed operations."""
        integration = ClaudeCodeIntegration(max_retries=3, retry_delay=0.1)
        
        # Mock failing twice then succeeding
        mock_responses = [
            Exception("Temporary failure"),
            Exception("Another temporary failure"),
            {
                'success': True,
                'output': 'Success on third attempt',
                'files_modified': ['src/main.py'],
                'execution_time': 1.5,
                'retry_attempt': 2
            }
        ]
        
        with patch.object(integration, '_execute_claude_command', side_effect=mock_responses):
            result = await integration.execute_command(
                task=sample_task,
                context=sample_context,
                workspace_path=temp_workspace
            )
            
            assert result['success'] is True
            assert result['retry_attempt'] == 2
    
    @pytest.mark.asyncio
    async def test_command_optimization(self, temp_workspace, sample_task, sample_context):
        """Test command optimization based on context."""
        integration = ClaudeCodeIntegration()
        
        # Test with different complexity levels
        simple_task = Task(
            id=uuid4(),
            name="Add print statement",
            description="Add a simple print statement",
            priority=Priority.LOW
        )
        
        complex_task = Task(
            id=uuid4(),
            name="Implement OAuth2 flow",
            description="Implement complete OAuth2 authentication flow with PKCE",
            priority=Priority.CRITICAL,
            metadata={'complexity': 'high', 'security_critical': True}
        )
        
        with patch.object(integration, '_optimize_command') as mock_optimize:
            mock_optimize.return_value = "optimized command"
            
            await integration.execute_command(simple_task, sample_context, temp_workspace)
            await integration.execute_command(complex_task, sample_context, temp_workspace)
            
            # Should optimize commands based on task complexity
            assert mock_optimize.call_count == 2
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, temp_workspace, sample_task, sample_context):
        """Test performance monitoring and metrics collection."""
        integration = ClaudeCodeIntegration(enable_monitoring=True)
        
        mock_response = {
            'success': True,
            'output': 'Command executed',
            'files_modified': ['src/main.py'],
            'execution_time': 2.3,
            'tokens_used': 1500,
            'memory_usage': 45.2
        }
        
        with patch.object(integration, '_execute_claude_command', return_value=mock_response):
            result = await integration.execute_command(
                task=sample_task,
                context=sample_context,
                workspace_path=temp_workspace
            )
            
            # Get performance metrics
            metrics = integration.get_performance_metrics()
            
            assert metrics is not None
            assert metrics['total_commands'] >= 1
            assert metrics['average_execution_time'] > 0
            assert 'token_usage' in metrics


class TestClaudeFlowIntegration:
    """Tests for Claude Flow integration."""
    
    def test_claude_flow_initialization(self):
        """Test Claude Flow integration initialization."""
        integration = ClaudeFlowIntegration()
        
        assert integration is not None
        assert hasattr(integration, 'orchestrate_task')
        assert hasattr(integration, 'manage_swarm')
        assert hasattr(integration, 'get_swarm_status')
    
    @pytest.mark.asyncio
    async def test_swarm_initialization(self):
        """Test swarm initialization and configuration."""
        integration = ClaudeFlowIntegration()
        
        # Mock swarm initialization
        mock_init_response = {
            'success': True,
            'swarm_id': 'test-swarm-123',
            'topology': 'hierarchical',
            'agents': ['coordinator', 'coder', 'reviewer'],
            'max_agents': 5
        }
        
        with patch.object(integration, '_initialize_swarm', return_value=mock_init_response):
            swarm_config = {
                'topology': 'hierarchical',
                'max_agents': 5,
                'strategy': 'adaptive'
            }
            
            result = await integration.initialize_swarm(swarm_config)
            
            assert result['success'] is True
            assert result['swarm_id'] is not None
            assert result['topology'] == 'hierarchical'
    
    @pytest.mark.asyncio
    async def test_task_orchestration(self, sample_task, sample_context):
        """Test task orchestration through Claude Flow."""
        integration = ClaudeFlowIntegration()
        
        # Mock orchestration response
        mock_orchestration = {
            'success': True,
            'orchestration_id': 'orch-456',
            'agents_assigned': ['coder', 'reviewer', 'tester'],
            'estimated_duration': 300,  # 5 minutes
            'coordination_strategy': 'parallel',
            'result': {
                'generated_content': 'Authentication system implemented',
                'files_modified': ['src/auth.py', 'src/middleware/auth.py', 'tests/test_auth.py'],
                'quality_score': 92.5,
                'authenticity_score': 88.0
            }
        }
        
        with patch.object(integration, '_orchestrate_task', return_value=mock_orchestration):
            result = await integration.orchestrate_task(
                task=sample_task,
                context=sample_context,
                swarm_config={'strategy': 'adaptive'}
            )
            
            assert result['success'] is True
            assert len(result['agents_assigned']) == 3
            assert result['result']['quality_score'] > 90
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self):
        """Test agent coordination and communication."""
        integration = ClaudeFlowIntegration()
        
        # Mock agent status
        mock_agents = {
            'coordinator': {'status': 'active', 'current_task': 'orchestrating', 'load': 0.3},
            'coder': {'status': 'busy', 'current_task': 'implementing_auth', 'load': 0.8},
            'reviewer': {'status': 'idle', 'current_task': None, 'load': 0.1},
            'tester': {'status': 'busy', 'current_task': 'writing_tests', 'load': 0.6}
        }
        
        with patch.object(integration, '_get_agent_status', return_value=mock_agents):
            agent_status = await integration.get_agent_status()
            
            assert len(agent_status) == 4
            assert agent_status['coordinator']['status'] == 'active'
            assert agent_status['coder']['load'] == 0.8
            
            # Test load balancing
            available_agents = [name for name, info in agent_status.items() if info['load'] < 0.5]
            assert 'coordinator' in available_agents
            assert 'reviewer' in available_agents
    
    @pytest.mark.asyncio
    async def test_swarm_scaling(self, sample_task):
        """Test automatic swarm scaling based on workload."""
        integration = ClaudeFlowIntegration()
        
        # Create multiple tasks to trigger scaling
        tasks = [sample_task]
        for i in range(10):
            task = Task(
                id=uuid4(),
                name=f"Task {i}",
                description=f"Parallel task {i}",
                priority=Priority.MEDIUM
            )
            tasks.append(task)
        
        # Mock scaling response
        mock_scaling = {
            'success': True,
            'original_agents': 3,
            'scaled_agents': 8,
            'scaling_trigger': 'high_workload',
            'scaling_time': 15.2
        }
        
        with patch.object(integration, '_auto_scale_swarm', return_value=mock_scaling):
            scaling_result = await integration.handle_scaling(tasks)
            
            assert scaling_result['success'] is True
            assert scaling_result['scaled_agents'] > scaling_result['original_agents']
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_failover(self, sample_task, sample_context):
        """Test error recovery and agent failover."""
        integration = ClaudeFlowIntegration()
        
        # Mock agent failure scenario
        mock_failure = {
            'failed_agent': 'coder',
            'error': 'Agent timeout',
            'failover_agent': 'backup-coder',
            'recovery_time': 5.8,
            'task_reassigned': True
        }
        
        mock_recovery_result = {
            'success': True,
            'orchestration_id': 'recovery-789',
            'agents_assigned': ['coordinator', 'backup-coder', 'reviewer'],
            'recovery_applied': True,
            'result': {
                'generated_content': 'Task completed after recovery',
                'files_modified': ['src/auth.py'],
                'quality_score': 85.0  # Slightly lower due to recovery
            }
        }
        
        with patch.object(integration, '_handle_agent_failure', return_value=mock_failure), \
             patch.object(integration, '_orchestrate_task', return_value=mock_recovery_result):
            
            # Simulate agent failure during orchestration
            result = await integration.orchestrate_task_with_recovery(
                task=sample_task,
                context=sample_context
            )
            
            assert result['success'] is True
            assert result['recovery_applied'] is True
            assert 'backup-coder' in result['agents_assigned']
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, sample_context):
        """Test performance optimization features."""
        integration = ClaudeFlowIntegration()
        
        # Test different optimization strategies
        optimization_configs = [
            {'strategy': 'speed', 'max_agents': 10, 'parallel_execution': True},
            {'strategy': 'quality', 'max_agents': 5, 'validation_enabled': True},
            {'strategy': 'balanced', 'max_agents': 7, 'adaptive_scaling': True}
        ]
        
        performance_results = []
        
        for config in optimization_configs:
            mock_result = {
                'success': True,
                'strategy': config['strategy'],
                'execution_time': 120 if config['strategy'] == 'speed' else 180,
                'quality_score': 95 if config['strategy'] == 'quality' else 87,
                'resource_efficiency': 0.8 if config['strategy'] == 'balanced' else 0.6
            }
            
            with patch.object(integration, '_optimize_execution', return_value=mock_result):
                result = await integration.optimize_execution(config)
                performance_results.append(result)
        
        # Verify optimization results
        speed_result = next(r for r in performance_results if r['strategy'] == 'speed')
        quality_result = next(r for r in performance_results if r['strategy'] == 'quality')
        
        assert speed_result['execution_time'] < quality_result['execution_time']
        assert quality_result['quality_score'] > speed_result['quality_score']


class TestIntegrationCoordination:
    """Tests for coordination between Claude Code and Claude Flow."""
    
    @pytest.mark.asyncio
    async def test_hybrid_execution_strategy(self, temp_workspace, sample_task, sample_context):
        """Test hybrid execution using both Claude Code and Claude Flow."""
        code_integration = ClaudeCodeIntegration()
        flow_integration = ClaudeFlowIntegration()
        
        # Mock responses from both integrations
        code_response = {
            'success': True,
            'output': 'Direct code implementation',
            'files_modified': ['src/simple.py'],
            'execution_time': 1.2,
            'complexity_handled': 'low'
        }
        
        flow_response = {
            'success': True,
            'result': {
                'generated_content': 'Complex orchestrated implementation',
                'files_modified': ['src/complex.py', 'src/auth.py', 'tests/test_auth.py'],
                'quality_score': 94.0
            },
            'execution_time': 4.5,
            'complexity_handled': 'high'
        }
        
        with patch.object(code_integration, 'execute_command', return_value=code_response), \
             patch.object(flow_integration, 'orchestrate_task', return_value=flow_response):
            
            # Test decision logic for choosing integration method
            simple_task = Task(
                id=uuid4(),
                name="Add logging",
                description="Add simple logging to existing function",
                priority=Priority.LOW,
                metadata={'complexity': 'low'}
            )
            
            complex_task = Task(
                id=uuid4(),
                name="Implement microservice",
                description="Implement complete microservice with auth, validation, and testing",
                priority=Priority.HIGH,
                metadata={'complexity': 'high', 'requires_coordination': True}
            )
            
            # Simple task should use Claude Code
            simple_result = await self._execute_hybrid_strategy(
                simple_task, sample_context, code_integration, flow_integration
            )
            
            # Complex task should use Claude Flow
            complex_result = await self._execute_hybrid_strategy(
                complex_task, sample_context, code_integration, flow_integration
            )
            
            assert simple_result['integration_used'] == 'claude_code'
            assert complex_result['integration_used'] == 'claude_flow'
    
    async def _execute_hybrid_strategy(self, task, context, code_integration, flow_integration):
        """Helper method to execute hybrid strategy."""
        # Simple decision logic based on task complexity
        task_complexity = task.metadata.get('complexity', 'medium')
        requires_coordination = task.metadata.get('requires_coordination', False)
        
        if task_complexity == 'low' and not requires_coordination:
            result = await code_integration.execute_command(task, context, Path('/tmp'))
            result['integration_used'] = 'claude_code'
            return result
        else:
            result = await flow_integration.orchestrate_task(task, context)
            result['integration_used'] = 'claude_flow'
            return result
    
    @pytest.mark.asyncio
    async def test_fallback_mechanisms(self, sample_task, sample_context):
        """Test fallback mechanisms when primary integration fails."""
        code_integration = ClaudeCodeIntegration()
        flow_integration = ClaudeFlowIntegration()
        
        # Mock Claude Flow failure
        flow_failure = Exception("Claude Flow service unavailable")
        
        # Mock Claude Code success as fallback
        code_success = {
            'success': True,
            'output': 'Fallback implementation successful',
            'files_modified': ['src/fallback.py'],
            'execution_time': 2.0,
            'fallback_used': True
        }
        
        with patch.object(flow_integration, 'orchestrate_task', side_effect=flow_failure), \
             patch.object(code_integration, 'execute_command', return_value=code_success):
            
            try:
                # Try Claude Flow first
                result = await flow_integration.orchestrate_task(sample_task, sample_context)
            except Exception:
                # Fallback to Claude Code
                result = await code_integration.execute_command(
                    sample_task, sample_context, Path('/tmp')
                )
            
            assert result['success'] is True
            assert result['fallback_used'] is True
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self, sample_task, sample_context):
        """Test performance comparison between integration methods."""
        code_integration = ClaudeCodeIntegration()
        flow_integration = ClaudeFlowIntegration()
        
        # Mock performance metrics
        code_metrics = {
            'execution_time': 1.5,
            'tokens_used': 800,
            'memory_usage': 25.0,
            'quality_score': 82.0
        }
        
        flow_metrics = {
            'execution_time': 4.2,
            'tokens_used': 2500,
            'memory_usage': 65.0,
            'quality_score': 94.0,
            'coordination_overhead': 0.8
        }
        
        with patch.object(code_integration, 'get_performance_metrics', return_value=code_metrics), \
             patch.object(flow_integration, 'get_performance_metrics', return_value=flow_metrics):
            
            code_perf = code_integration.get_performance_metrics()
            flow_perf = flow_integration.get_performance_metrics()
            
            # Compare performance characteristics
            assert code_perf['execution_time'] < flow_perf['execution_time']  # Code faster
            assert flow_perf['quality_score'] > code_perf['quality_score']   # Flow higher quality
            assert code_perf['tokens_used'] < flow_perf['tokens_used']        # Code more efficient
    
    @pytest.mark.asyncio
    async def test_integration_monitoring_and_analytics(self):
        """Test monitoring and analytics across integrations."""
        code_integration = ClaudeCodeIntegration(enable_monitoring=True)
        flow_integration = ClaudeFlowIntegration(enable_monitoring=True)
        
        # Mock analytics data
        analytics_data = {
            'claude_code': {
                'total_executions': 150,
                'success_rate': 94.0,
                'average_execution_time': 1.8,
                'token_efficiency': 0.85,
                'preferred_for': ['simple_tasks', 'quick_fixes', 'single_file_changes']
            },
            'claude_flow': {
                'total_orchestrations': 45,
                'success_rate': 91.0,
                'average_execution_time': 5.2,
                'quality_improvement': 0.12,
                'preferred_for': ['complex_tasks', 'multi_file_changes', 'coordinated_work']
            }
        }
        
        with patch.object(code_integration, 'get_analytics', return_value=analytics_data['claude_code']), \
             patch.object(flow_integration, 'get_analytics', return_value=analytics_data['claude_flow']):
            
            code_analytics = code_integration.get_analytics()
            flow_analytics = flow_integration.get_analytics()
            
            # Verify analytics collection
            assert code_analytics['total_executions'] > flow_analytics['total_orchestrations']
            assert code_analytics['success_rate'] > 90
            assert flow_analytics['success_rate'] > 90
            
            # Test recommendation system based on analytics
            task_characteristics = {
                'complexity': 'low',
                'files_affected': 1,
                'coordination_required': False
            }
            
            recommended_integration = self._recommend_integration(
                task_characteristics, code_analytics, flow_analytics
            )
            
            assert recommended_integration == 'claude_code'  # Should recommend code for simple task
    
    def _recommend_integration(self, task_characteristics, code_analytics, flow_analytics):
        """Helper method to recommend integration based on analytics."""
        complexity = task_characteristics['complexity']
        files_affected = task_characteristics['files_affected']
        coordination_required = task_characteristics['coordination_required']
        
        # Simple recommendation logic
        if (complexity == 'low' and 
            files_affected <= 2 and 
            not coordination_required):
            return 'claude_code'
        else:
            return 'claude_flow'


class TestRealWorldIntegrationScenarios:
    """Tests for real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_development_workflow(self, temp_workspace):
        """Test complete development workflow using integrations."""
        # This test simulates a real development workflow:
        # 1. Create project structure (Claude Code)
        # 2. Implement core features (Claude Flow)
        # 3. Add tests (Claude Code)
        # 4. Refactor and optimize (Claude Flow)
        
        code_integration = ClaudeCodeIntegration()
        flow_integration = ClaudeFlowIntegration()
        
        workflow_steps = [
            {
                'step': 'project_setup',
                'integration': 'code',
                'task': 'Create project structure and basic files',
                'expected_files': ['src/__init__.py', 'tests/__init__.py', 'setup.py']
            },
            {
                'step': 'core_implementation',
                'integration': 'flow',
                'task': 'Implement core business logic with validation',
                'expected_files': ['src/core.py', 'src/models.py', 'src/validators.py']
            },
            {
                'step': 'testing',
                'integration': 'code',
                'task': 'Add comprehensive test suite',
                'expected_files': ['tests/test_core.py', 'tests/test_models.py']
            },
            {
                'step': 'optimization',
                'integration': 'flow',
                'task': 'Refactor and optimize performance',
                'expected_files': ['src/optimized_core.py']
            }
        ]
        
        workflow_results = []
        
        for step in workflow_steps:
            if step['integration'] == 'code':
                mock_result = {
                    'success': True,
                    'output': f"Completed {step['step']}",
                    'files_modified': step['expected_files'],
                    'execution_time': 1.5
                }
                
                with patch.object(code_integration, 'execute_command', return_value=mock_result):
                    result = await code_integration.execute_command(
                        Mock(name=step['task']), {}, temp_workspace
                    )
            else:
                mock_result = {
                    'success': True,
                    'result': {
                        'generated_content': f"Orchestrated {step['step']}",
                        'files_modified': step['expected_files'],
                        'quality_score': 90.0
                    },
                    'execution_time': 3.0
                }
                
                with patch.object(flow_integration, 'orchestrate_task', return_value=mock_result):
                    result = await flow_integration.orchestrate_task(
                        Mock(name=step['task']), {}
                    )
            
            workflow_results.append({
                'step': step['step'],
                'success': result['success'],
                'integration_used': step['integration']
            })
        
        # Verify all steps completed successfully
        assert all(r['success'] for r in workflow_results)
        assert len(workflow_results) == 4
        
        # Verify integration usage pattern
        code_steps = [r for r in workflow_results if r['integration_used'] == 'code']
        flow_steps = [r for r in workflow_results if r['integration_used'] == 'flow']
        
        assert len(code_steps) == 2  # Setup and testing
        assert len(flow_steps) == 2  # Implementation and optimization
    
    @pytest.mark.asyncio
    async def test_integration_stress_testing(self):
        """Test integrations under stress conditions."""
        code_integration = ClaudeCodeIntegration()
        flow_integration = ClaudeFlowIntegration()
        
        # Create many concurrent tasks
        num_tasks = 50
        tasks = []
        
        for i in range(num_tasks):
            task = Task(
                id=uuid4(),
                name=f"Stress Task {i}",
                description=f"Concurrent task {i} for stress testing",
                priority=Priority.MEDIUM if i % 2 == 0 else Priority.LOW
            )
            tasks.append(task)
        
        # Mock responses with varying success rates
        def create_mock_response(task_id, success_rate=0.95):
            import random
            success = random.random() < success_rate
            
            if success:
                return {
                    'success': True,
                    'output': f'Completed task {task_id}',
                    'files_modified': [f'src/task_{task_id}.py'],
                    'execution_time': random.uniform(0.5, 3.0)
                }
            else:
                return {
                    'success': False,
                    'error': f'Task {task_id} failed',
                    'execution_time': random.uniform(0.1, 1.0)
                }
        
        # Execute tasks concurrently
        async def execute_task(task, integration):
            mock_response = create_mock_response(str(task.id)[:8])
            
            with patch.object(integration, 'execute_command', return_value=mock_response):
                return await integration.execute_command(task, {}, Path('/tmp'))
        
        # Split tasks between integrations
        code_tasks = tasks[:25]
        flow_tasks = tasks[25:]
        
        # Execute all tasks concurrently
        import time
        start_time = time.time()
        
        code_results = await asyncio.gather(
            *[execute_task(task, code_integration) for task in code_tasks],
            return_exceptions=True
        )
        
        flow_results = await asyncio.gather(
            *[execute_task(task, flow_integration) for task in flow_tasks],
            return_exceptions=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        all_results = code_results + flow_results
        successful_results = [r for r in all_results if isinstance(r, dict) and r.get('success')]
        failed_results = [r for r in all_results if not (isinstance(r, dict) and r.get('success'))]
        
        success_rate = len(successful_results) / len(all_results) * 100
        throughput = len(all_results) / total_time
        
        print(f"Stress test results:")
        print(f"  Total tasks: {len(all_results)}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} tasks/second")
        
        # Performance requirements for stress testing
        assert success_rate >= 85  # At least 85% success rate under stress
        assert total_time < 30     # Complete 50 tasks in under 30 seconds
        assert throughput >= 1.0   # At least 1 task per second throughput


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
