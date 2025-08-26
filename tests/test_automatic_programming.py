#!/usr/bin/env python3
"""
Tests for the Automatic Programming Pipeline

Tests the core components of the automatic programming system:
- RequirementsAnalyzer
- TaskDecomposer  
- CodeGenerator
- ValidationEngine
- AutomaticProgrammingCoordinator
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claude_tui.automation import (
    AutomaticProgrammingCoordinator,
    RequirementsAnalyzer,
    TaskDecomposer, 
    CodeGenerator,
    ValidationEngine,
    RequirementsAnalysis,
    TaskComponent,
    PipelineStatus
)
from claude_tui.core.config_manager import ConfigManager
from claude_tui.integrations.claude_code_client import ClaudeCodeClient
from claude_tui.integrations.claude_flow_client import ClaudeFlowClient


class TestRequirementsAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test Requirements Analyzer component."""
    
    async def asyncSetUp(self):
        self.config_manager = MagicMock()
        self.claude_code_client = AsyncMock(spec=ClaudeCodeClient)
        self.analyzer = RequirementsAnalyzer(self.claude_code_client)
    
    async def test_analyze_success(self):
        """Test successful requirements analysis."""
        # Mock successful AI response
        self.claude_code_client.execute_task.return_value = {
            'success': True,
            'analysis': {
                'project_type': 'web_app',
                'complexity_score': 7,
                'architecture_pattern': 'microservices',
                'technologies': ['Python', 'FastAPI', 'PostgreSQL'],
                'quality_attributes': ['scalability', 'security'],
                'constraints': ['time', 'budget'],
                'assumptions': ['cloud_deployment'],
                'risks': ['complexity']
            }
        }
        
        requirements = "Create a scalable web application for user management"
        
        result = await self.analyzer.analyze(requirements)
        
        self.assertIsInstance(result, RequirementsAnalysis)
        self.assertEqual(result.project_type, 'web_app')
        self.assertEqual(result.estimated_complexity, 7)
        self.assertEqual(result.recommended_architecture, 'microservices')
        self.assertIn('Python', result.suggested_technologies)
        
    async def test_analyze_fallback(self):
        """Test fallback analysis when AI analysis fails."""
        # Mock failed AI response
        self.claude_code_client.execute_task.return_value = {
            'success': False,
            'error': 'API Error'
        }
        
        requirements = "Create a simple calculator"
        
        result = await self.analyzer.analyze(requirements)
        
        self.assertIsInstance(result, RequirementsAnalysis)
        self.assertEqual(result.project_type, 'general')
        self.assertEqual(result.estimated_complexity, 5)
        self.assertEqual(result.recommended_architecture, 'modular')


class TestTaskDecomposer(unittest.IsolatedAsyncioTestCase):
    """Test Task Decomposer component."""
    
    async def asyncSetUp(self):
        self.claude_code_client = AsyncMock(spec=ClaudeCodeClient)
        self.decomposer = TaskDecomposer(self.claude_code_client)
        
        # Create sample requirements analysis
        self.requirements_analysis = RequirementsAnalysis(
            original_requirements="Create a REST API",
            parsed_requirements={},
            project_type="web_app",
            estimated_complexity=6,
            recommended_architecture="layered",
            suggested_technologies=["Python", "FastAPI"],
            quality_attributes=["performance"],
            constraints=[],
            assumptions=[],
            risks=[]
        )
    
    async def test_decompose_success(self):
        """Test successful task decomposition."""
        # Mock successful AI response
        self.claude_code_client.execute_task.return_value = {
            'success': True,
            'tasks': [
                {
                    'id': 'setup_project',
                    'description': 'Set up FastAPI project structure',
                    'file_path': 'main.py',
                    'agent_type': 'coder',
                    'priority': 1,
                    'complexity': 2,
                    'dependencies': [],
                    'requirements': ['FastAPI installation'],
                    'validation_criteria': ['Project runs without errors']
                },
                {
                    'id': 'implement_api',
                    'description': 'Implement REST endpoints',
                    'file_path': 'api/routes.py',
                    'agent_type': 'coder',
                    'priority': 5,
                    'complexity': 7,
                    'dependencies': ['setup_project'],
                    'requirements': ['CRUD operations'],
                    'validation_criteria': ['All endpoints respond correctly']
                }
            ]
        }
        
        result = await self.decomposer.decompose(self.requirements_analysis)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Check first task
        task1 = result[0]
        self.assertEqual(task1.id, 'setup_project')
        self.assertEqual(task1.description, 'Set up FastAPI project structure')
        self.assertEqual(task1.agent_type, 'coder')
        self.assertEqual(task1.priority, 1)
        
        # Check dependencies
        task2 = result[1]
        self.assertIn('setup_project', task2.dependencies)
    
    async def test_decompose_fallback(self):
        """Test fallback task creation when decomposition fails."""
        # Mock failed AI response
        self.claude_code_client.execute_task.return_value = {
            'success': False,
            'error': 'Decomposition failed'
        }
        
        result = await self.decomposer.decompose(self.requirements_analysis)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check that fallback tasks are created
        task_ids = [task.id for task in result]
        self.assertIn('setup_project', task_ids)
        self.assertIn('implement_core', task_ids)


class TestCodeGenerator(unittest.IsolatedAsyncioTestCase):
    """Test Code Generator component."""
    
    async def asyncSetUp(self):
        self.claude_code_client = AsyncMock(spec=ClaudeCodeClient)
        self.claude_flow_client = AsyncMock(spec=ClaudeFlowClient)
        self.generator = CodeGenerator(self.claude_code_client, self.claude_flow_client)
        
        # Mock swarm initialization
        self.claude_flow_client.initialize_swarm.return_value = "swarm_123"
        self.claude_flow_client.shutdown_swarm.return_value = True
        
        # Create sample task components
        self.task_components = [
            TaskComponent(
                id="task1",
                description="Create main module",
                file_path="main.py",
                agent_type="coder",
                priority=1,
                estimated_complexity=3
            ),
            TaskComponent(
                id="task2", 
                description="Add tests",
                file_path="test_main.py",
                dependencies=["task1"],
                agent_type="tester",
                priority=5,
                estimated_complexity=2
            )
        ]
        
        self.requirements_analysis = RequirementsAnalysis(
            original_requirements="Create a Python module",
            parsed_requirements={},
            project_type="library",
            estimated_complexity=4,
            recommended_architecture="modular",
            suggested_technologies=["Python"],
            quality_attributes=[],
            constraints=[],
            assumptions=[],
            risks=[]
        )
    
    async def test_generate_code_success(self):
        """Test successful code generation."""
        # Mock task execution responses
        self.claude_code_client.execute_task.side_effect = [
            {
                'success': True,
                'generated_files': ['main.py'],
                'execution_time': 2.5,
                'validation_passed': True
            },
            {
                'success': True,
                'generated_files': ['test_main.py'],
                'execution_time': 1.8,
                'validation_passed': True
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.generator.generate_code(
                self.task_components, temp_dir, self.requirements_analysis
            )
            
            self.assertTrue(result['success'])
            self.assertIn('generated_files', result)
            self.assertIn('agent_reports', result)
            self.assertGreater(result['execution_time'], 0)
            
            # Verify swarm operations were called
            self.claude_flow_client.initialize_swarm.assert_called_once()
            self.claude_flow_client.shutdown_swarm.assert_called_once()
    
    async def test_task_dependency_grouping(self):
        """Test that tasks are properly grouped by dependencies."""
        groups = self.generator._group_tasks_by_dependencies(self.task_components)
        
        self.assertEqual(len(groups), 2)  # Two dependency levels
        
        # First group should have task1 (no dependencies)
        first_group = groups[0]
        self.assertEqual(len(first_group), 1)
        self.assertEqual(first_group[0].id, "task1")
        
        # Second group should have task2 (depends on task1)  
        second_group = groups[1]
        self.assertEqual(len(second_group), 1)
        self.assertEqual(second_group[0].id, "task2")


class TestValidationEngine(unittest.IsolatedAsyncioTestCase):
    """Test Validation Engine component."""
    
    async def asyncSetUp(self):
        self.claude_code_client = AsyncMock(spec=ClaudeCodeClient)
        self.engine = ValidationEngine(self.claude_code_client)
        
        self.requirements_analysis = RequirementsAnalysis(
            original_requirements="Create a calculator",
            parsed_requirements={},
            project_type="utility",
            estimated_complexity=3,
            recommended_architecture="simple",
            suggested_technologies=["Python"],
            quality_attributes=[],
            constraints=[],
            assumptions=[],
            risks=[]
        )
    
    async def test_validate_results_success(self):
        """Test successful validation of results."""
        generation_results = {
            'success': True,
            'generated_files': ['calculator.py'],
            'agent_reports': {}
        }
        
        # Mock file validation
        with patch.object(self.engine, '_validate_file') as mock_validate_file:
            mock_validate_file.return_value = {
                'file_path': 'calculator.py',
                'quality_score': 0.85,
                'issues': [],
                'suggestions': ['Add more comments'],
                'syntax_valid': True
            }
            
            # Mock requirements validation
            self.claude_code_client.execute_task.return_value = {
                'success': True,
                'assessment': {
                    'coverage_percentage': 90.0,
                    'completeness_score': 0.9,
                    'missing_features': [],
                    'alignment_score': 0.85
                }
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await self.engine.validate_results(
                    generation_results, temp_dir, self.requirements_analysis
                )
                
                self.assertTrue(result['overall_success'])
                self.assertEqual(result['quality_score'], 0.85)
                self.assertEqual(result['requirements_coverage'], 90.0)
                self.assertGreater(result['validation_time'], 0)


class TestAutomaticProgrammingCoordinator(unittest.IsolatedAsyncioTestCase):
    """Test the main Automatic Programming Coordinator."""
    
    async def asyncSetUp(self):
        self.config_manager = MagicMock(spec=ConfigManager)
        
        # Mock client initialization
        with patch('claude_tui.automation.automatic_programming.ClaudeCodeClient') as mock_code_client, \
             patch('claude_tui.automation.automatic_programming.ClaudeFlowClient') as mock_flow_client:
            
            self.coordinator = AutomaticProgrammingCoordinator(self.config_manager)
            self.coordinator.claude_code_client = AsyncMock()
            self.coordinator.claude_flow_client = AsyncMock()
            
            # Mock client health checks
            self.coordinator.claude_code_client.health_check.return_value = True
            self.coordinator.claude_flow_client.initialize = AsyncMock()
    
    async def test_initialization(self):
        """Test coordinator initialization."""
        await self.coordinator.initialize()
        
        self.coordinator.claude_flow_client.initialize.assert_called_once()
        self.coordinator.claude_code_client.health_check.assert_called_once()
    
    async def test_pipeline_status_tracking(self):
        """Test pipeline status tracking."""
        # Initial status should be INITIALIZING
        self.assertEqual(self.coordinator.current_status, PipelineStatus.INITIALIZING)
        
        # Get status
        status = await self.coordinator.get_pipeline_status()
        
        self.assertIn('pipeline_id', status)
        self.assertIn('status', status)
        self.assertEqual(status['status'], PipelineStatus.INITIALIZING.value)
    
    async def test_memory_context_operations(self):
        """Test memory context storage and retrieval."""
        test_key = 'test_context'
        test_value = {'data': 'test'}
        
        # Set context
        await self.coordinator.set_memory_context(test_key, test_value)
        
        # Get context
        retrieved_value = await self.coordinator.get_memory_context(test_key)
        
        self.assertEqual(retrieved_value, test_value)
        
        # Test non-existent key
        none_value = await self.coordinator.get_memory_context('non_existent')
        self.assertIsNone(none_value)
    
    @patch('claude_tui.automation.automatic_programming.Path')
    async def test_generate_code_pipeline(self, mock_path):
        """Test the complete code generation pipeline."""
        # Mock Path operations
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.mkdir = MagicMock()
        
        # Mock pipeline components
        with patch.object(self.coordinator.requirements_analyzer, 'analyze') as mock_analyze, \
             patch.object(self.coordinator.task_decomposer, 'decompose') as mock_decompose, \
             patch.object(self.coordinator.code_generator, 'generate_code') as mock_generate, \
             patch.object(self.coordinator.validation_engine, 'validate_results') as mock_validate:
            
            # Set up mock returns
            mock_analyze.return_value = RequirementsAnalysis(
                original_requirements="Test requirements",
                parsed_requirements={},
                project_type="test",
                estimated_complexity=3,
                recommended_architecture="simple",
                suggested_technologies=["Python"],
                quality_attributes=[],
                constraints=[],
                assumptions=[],
                risks=[]
            )
            
            mock_decompose.return_value = [
                TaskComponent(
                    id="test_task",
                    description="Test task",
                    agent_type="coder",
                    priority=1,
                    estimated_complexity=2
                )
            ]
            
            mock_generate.return_value = {
                'success': True,
                'generated_files': ['test.py'],
                'agent_reports': {},
                'execution_time': 1.0
            }
            
            mock_validate.return_value = {
                'overall_success': True,
                'quality_score': 0.8,
                'validation_time': 0.5
            }
            
            # Run pipeline
            result = await self.coordinator.generate_code(
                requirements="Create a test application",
                project_path="/tmp/test_project"
            )
            
            # Verify result
            self.assertTrue(result.success)
            self.assertIsNotNone(result.requirements_analysis)
            self.assertGreater(len(result.task_components), 0)
            self.assertGreater(result.execution_time, 0)
            self.assertEqual(result.quality_score, 0.8)
            
            # Verify pipeline methods were called
            mock_analyze.assert_called_once()
            mock_decompose.assert_called_once()
            mock_generate.assert_called_once()
            mock_validate.assert_called_once()


class TestIntegrationPoints(unittest.IsolatedAsyncioTestCase):
    """Test integration between components."""
    
    async def test_end_to_end_mock_integration(self):
        """Test end-to-end integration with mocked external dependencies."""
        
        with patch('claude_tui.automation.automatic_programming.ClaudeCodeClient') as mock_code_client, \
             patch('claude_tui.automation.automatic_programming.ClaudeFlowClient') as mock_flow_client:
            
            # Set up mock clients
            code_client_instance = AsyncMock()
            flow_client_instance = AsyncMock()
            
            mock_code_client.return_value = code_client_instance
            mock_flow_client.return_value = flow_client_instance
            
            # Configure mock responses
            code_client_instance.health_check.return_value = True
            code_client_instance.execute_task.return_value = {
                'success': True,
                'analysis': {
                    'project_type': 'utility',
                    'complexity_score': 4,
                    'architecture_pattern': 'simple',
                    'technologies': ['Python']
                },
                'tasks': [
                    {
                        'id': 'main_task',
                        'description': 'Create main functionality',
                        'file_path': 'main.py',
                        'agent_type': 'coder',
                        'priority': 1,
                        'complexity': 3
                    }
                ],
                'generated_files': ['main.py'],
                'validation_passed': True
            }
            
            flow_client_instance.initialize = AsyncMock()
            flow_client_instance.initialize_swarm.return_value = "test_swarm"
            flow_client_instance.shutdown_swarm.return_value = True
            
            # Create coordinator and test
            config_manager = MagicMock()
            coordinator = AutomaticProgrammingCoordinator(config_manager)
            
            await coordinator.initialize()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await coordinator.generate_code(
                    requirements="Create a simple calculator",
                    project_path=temp_dir
                )
                
                # Verify integration worked
                self.assertIsNotNone(result)
                self.assertTrue(hasattr(result, 'success'))
                self.assertTrue(hasattr(result, 'execution_time'))


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main()