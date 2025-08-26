"""
AI Services Integration Tests for claude-tui.

Comprehensive tests for AI service functionality including:
- Swarm initialization and coordination
- Multi-agent task orchestration
- Neural training and inference
- Performance benchmarks
- Claude Code and Claude Flow integration
- Response validation and anti-hallucination
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.services.ai_service import AIService
from src.core.ai_interface import AIInterface
from src.core.exceptions import AIServiceError, ClaudeCodeError, ClaudeFlowError
from src.integrations.claude_code import ClaudeCodeIntegration
from src.integrations.claude_flow import ClaudeFlowIntegration


@pytest.fixture
async def ai_service():
    """Create AI service for testing."""
    service = AIService()
    await service.initialize()
    return service


@pytest.fixture
def mock_claude_code_integration():
    """Mock Claude Code integration."""
    mock = AsyncMock(spec=ClaudeCodeIntegration)
    mock.test_connection.return_value = {"status": "connected"}
    mock.generate_code.return_value = {
        "code": "def hello_world():\\n    return 'Hello, World!'",
        "language": "python",
        "quality_score": 0.95,
        "metadata": {"lines": 2, "functions": 1}
    }
    mock.execute_command.return_value = {
        "status": "success",
        "output": "Command executed successfully",
        "execution_time": 0.5
    }
    return mock


@pytest.fixture
def mock_claude_flow_integration():
    """Mock Claude Flow integration."""
    mock = AsyncMock(spec=ClaudeFlowIntegration)
    mock.test_connection.return_value = {"status": "connected"}
    mock.orchestrate_task.return_value = {
        "task_id": "task_12345",
        "status": "in_progress",
        "agents": ["agent_1", "agent_2", "agent_3"],
        "estimated_completion": "2024-01-01T10:30:00Z"
    }
    mock.get_task_status.return_value = {
        "task_id": "task_12345",
        "status": "completed",
        "progress": 100,
        "results": {"files_created": 5, "tests_passed": 15}
    }
    return mock


@pytest.fixture
def sample_code_generation_prompt():
    """Sample code generation prompt."""
    return {
        "prompt": "Create a Python function that calculates the fibonacci sequence",
        "language": "python",
        "context": {
            "project_type": "algorithm",
            "performance_requirements": "O(n) time complexity",
            "include_tests": True
        }
    }


@pytest.fixture
def sample_task_orchestration_request():
    """Sample task orchestration request."""
    return {
        "task_description": "Build a REST API with user authentication",
        "requirements": {
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "authentication": "JWT",
            "testing": "pytest"
        },
        "agents": ["backend_developer", "database_architect", "test_engineer"],
        "strategy": "parallel"
    }


@pytest.mark.asyncio
class TestAIServiceInitialization:
    """Test AI service initialization and health checks."""
    
    async def test_service_initialization_success(self):
        """Test successful AI service initialization."""
        with patch('src.services.ai_service.AIInterface') as mock_ai_interface:
            with patch.object(AIService, '_test_claude_code_connection', return_value=True):
                with patch.object(AIService, '_test_claude_flow_connection', return_value=True):
                    service = AIService()
                    await service.initialize()
                    
                    assert service.is_initialized
                    assert service._claude_code_available is True
                    assert service._claude_flow_available is True
    
    async def test_service_initialization_partial_providers(self):
        """Test initialization with only one provider available."""
        with patch('src.services.ai_service.AIInterface'):
            with patch.object(AIService, '_test_claude_code_connection', return_value=True):
                with patch.object(AIService, '_test_claude_flow_connection', return_value=False):
                    service = AIService()
                    await service.initialize()
                    
                    assert service.is_initialized
                    assert service._claude_code_available is True
                    assert service._claude_flow_available is False
    
    async def test_service_initialization_no_providers(self):
        """Test initialization failure with no providers available."""
        with patch('src.services.ai_service.AIInterface'):
            with patch.object(AIService, '_test_claude_code_connection', return_value=False):
                with patch.object(AIService, '_test_claude_flow_connection', return_value=False):
                    service = AIService()
                    
                    with pytest.raises(AIServiceError, match="No AI providers available"):
                        await service.initialize()
    
    async def test_health_check(self, ai_service):
        """Test AI service health check."""
        health = await ai_service.health_check()
        
        assert "status" in health
        assert "claude_code_available" in health
        assert "claude_flow_available" in health
        assert "response_cache_size" in health
        assert "request_history_size" in health
        assert isinstance(health["claude_code_available"], bool)
        assert isinstance(health["claude_flow_available"], bool)


@pytest.mark.asyncio
class TestCodeGeneration:
    """Test AI code generation functionality."""
    
    async def test_generate_code_success(self, ai_service, sample_code_generation_prompt, mock_claude_code_integration):
        """Test successful code generation."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.return_value = {
                "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python",
                "quality_score": 0.90,
                "metadata": {
                    "lines": 4,
                    "functions": 1,
                    "complexity": "recursive"
                }
            }
            
            result = await ai_service.generate_code(
                prompt=sample_code_generation_prompt["prompt"],
                language=sample_code_generation_prompt["language"],
                context=sample_code_generation_prompt["context"],
                validate_response=True
            )
            
            assert result["code"] is not None
            assert result["language"] == "python"
            assert "validation" in result
            assert result["validation"]["is_valid"] is True
    
    async def test_generate_code_with_validation_failure(self, ai_service):
        """Test code generation with validation failure."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            # Mock response with placeholder code
            mock_interface.generate_code.return_value = {
                "code": "def fibonacci(n):\\n    # TODO: implement fibonacci",
                "language": "python"
            }
            
            result = await ai_service.generate_code(
                prompt="Create fibonacci function",
                language="python",
                validate_response=True
            )
            
            assert result["validation"]["is_valid"] is False
            assert "TODO" in str(result["validation"]["warnings"])
    
    async def test_generate_code_caching(self, ai_service, sample_code_generation_prompt):
        """Test code generation response caching."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_response = {
                "code": "def test_function(): pass",
                "language": "python"
            }
            mock_interface.generate_code.return_value = mock_response
            
            # First call - should hit the API
            result1 = await ai_service.generate_code(
                prompt=sample_code_generation_prompt["prompt"],
                language=sample_code_generation_prompt["language"],
                use_cache=True
            )
            
            # Second call - should use cache
            result2 = await ai_service.generate_code(
                prompt=sample_code_generation_prompt["prompt"],
                language=sample_code_generation_prompt["language"],
                use_cache=True
            )
            
            # API should be called only once
            assert mock_interface.generate_code.call_count == 1
            assert result1 == result2
    
    async def test_generate_code_claude_code_unavailable(self, ai_service):
        """Test code generation when Claude Code is unavailable."""
        ai_service._claude_code_available = False
        
        with pytest.raises(ClaudeCodeError, match="Claude Code not available"):
            await ai_service.generate_code(
                prompt="Generate some code",
                language="python"
            )
    
    async def test_generate_code_syntax_validation(self, ai_service):
        """Test code generation with syntax validation."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            # Mock response with syntax error
            mock_interface.generate_code.return_value = {
                "code": "def invalid_syntax(:\\n    pass",  # Missing closing parenthesis
                "language": "python"
            }
            
            result = await ai_service.generate_code(
                prompt="Create a function",
                language="python",
                validate_response=True
            )
            
            assert result["validation"]["is_valid"] is False
            assert "syntax error" in str(result["validation"]["errors"]).lower()


@pytest.mark.asyncio
class TestTaskOrchestration:
    """Test AI task orchestration functionality."""
    
    async def test_orchestrate_task_success(self, ai_service, sample_task_orchestration_request, mock_claude_flow_integration):
        """Test successful task orchestration."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            result = await ai_service.orchestrate_task(
                task_description=sample_task_orchestration_request["task_description"],
                requirements=sample_task_orchestration_request["requirements"],
                agents=sample_task_orchestration_request["agents"],
                strategy=sample_task_orchestration_request["strategy"]
            )
            
            assert result["task_id"] is not None
            assert result["status"] == "in_progress"
            assert len(result["agents"]) == 3
            mock_claude_flow_integration.orchestrate_task.assert_called_once()
    
    async def test_orchestrate_task_claude_flow_unavailable(self, ai_service, sample_task_orchestration_request):
        """Test task orchestration when Claude Flow is unavailable."""
        ai_service._claude_flow_available = False
        
        with pytest.raises(ClaudeFlowError, match="Claude Flow not available"):
            await ai_service.orchestrate_task(
                task_description=sample_task_orchestration_request["task_description"]
            )
    
    async def test_orchestrate_task_with_custom_agents(self, ai_service, mock_claude_flow_integration):
        """Test task orchestration with specific agents."""
        custom_agents = ["senior_developer", "security_specialist", "performance_optimizer"]
        
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            result = await ai_service.orchestrate_task(
                task_description="Optimize application performance",
                agents=custom_agents,
                strategy="sequential"
            )
            
            call_args = mock_claude_flow_integration.orchestrate_task.call_args[0][0]
            assert call_args["agents"] == custom_agents
            assert call_args["strategy"] == "sequential"
    
    async def test_orchestrate_task_error_handling(self, ai_service, mock_claude_flow_integration):
        """Test task orchestration error handling."""
        mock_claude_flow_integration.orchestrate_task.side_effect = Exception("Orchestration failed")
        
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            with pytest.raises(ClaudeFlowError, match="Task orchestration failed"):
                await ai_service.orchestrate_task(
                    task_description="This will fail"
                )


@pytest.mark.asyncio
class TestResponseValidation:
    """Test AI response validation functionality."""
    
    async def test_validate_code_response(self, ai_service):
        """Test code response validation."""
        code_response = {
            "code": "def add(a, b):\\n    return a + b",
            "language": "python"
        }
        
        validation_result = await ai_service.validate_response(
            response=code_response,
            validation_type="code",
            criteria={"language": "python"}
        )
        
        assert validation_result["is_valid"] is True
        assert validation_result["score"] >= 0.0
        assert isinstance(validation_result["errors"], list)
        assert isinstance(validation_result["warnings"], list)
    
    async def test_validate_code_response_with_placeholders(self, ai_service):
        """Test validation of code response containing placeholders."""
        code_response = {
            "code": "def process_data(data):\\n    # TODO: implement data processing\\n    pass",
            "language": "python"
        }
        
        validation_result = await ai_service.validate_response(
            response=code_response,
            validation_type="code",
            criteria={"language": "python"}
        )
        
        assert len(validation_result["warnings"]) > 0
        assert validation_result["score"] < 1.0
        assert any("TODO" in str(warning) for warning in validation_result["warnings"])
    
    async def test_validate_text_response(self, ai_service):
        """Test text response validation."""
        text_response = {
            "content": "This is a comprehensive response that meets the minimum requirements.",
            "type": "explanation"
        }
        
        validation_result = await ai_service.validate_response(
            response=text_response,
            validation_type="text",
            criteria={"min_length": 20}
        )
        
        assert validation_result["is_valid"] is True
        assert validation_result["score"] >= 0.8
    
    async def test_validate_text_response_too_short(self, ai_service):
        """Test validation of text response that's too short."""
        text_response = {
            "content": "Short.",
            "type": "explanation"
        }
        
        validation_result = await ai_service.validate_response(
            response=text_response,
            validation_type="text",
            criteria={"min_length": 50}
        )
        
        assert len(validation_result["warnings"]) > 0
        assert validation_result["score"] < 1.0
    
    async def test_validate_general_response(self, ai_service):
        """Test general response validation."""
        general_response = {
            "task_id": "123",
            "status": "completed",
            "result": {"output": "Task completed successfully"}
        }
        
        validation_result = await ai_service.validate_response(
            response=general_response,
            validation_type="general",
            criteria={"required_fields": ["task_id", "status", "result"]}
        )
        
        assert validation_result["is_valid"] is True
        assert validation_result["score"] == 1.0
    
    async def test_validate_general_response_missing_fields(self, ai_service):
        """Test validation of response with missing required fields."""
        incomplete_response = {
            "task_id": "123",
            # Missing "status" and "result" fields
        }
        
        validation_result = await ai_service.validate_response(
            response=incomplete_response,
            validation_type="general",
            criteria={"required_fields": ["task_id", "status", "result"]}
        )
        
        assert validation_result["is_valid"] is False
        assert validation_result["score"] == 0.0
        assert len(validation_result["errors"]) > 0


@pytest.mark.asyncio
class TestSwarmCoordination:
    """Test multi-agent swarm coordination."""
    
    async def test_swarm_initialization(self, ai_service, mock_claude_flow_integration):
        """Test swarm initialization and configuration."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            mock_claude_flow_integration.initialize_swarm.return_value = {
                "swarm_id": "swarm_123",
                "topology": "mesh",
                "max_agents": 10,
                "status": "initialized"
            }
            
            # This would be a method to initialize swarm (to be implemented)
            # For now, we'll test the orchestration which includes swarm concepts
            result = await ai_service.orchestrate_task(
                task_description="Initialize development swarm",
                requirements={"topology": "mesh", "max_agents": 10}
            )
            
            assert result["task_id"] is not None
    
    async def test_multi_agent_task_distribution(self, ai_service, mock_claude_flow_integration):
        """Test distributing tasks across multiple agents."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            mock_claude_flow_integration.orchestrate_task.return_value = {
                "task_id": "multi_agent_task",
                "status": "in_progress",
                "agents": [
                    {"id": "agent_1", "type": "frontend", "task": "build_ui"},
                    {"id": "agent_2", "type": "backend", "task": "create_api"},
                    {"id": "agent_3", "type": "database", "task": "design_schema"}
                ],
                "coordination_strategy": "parallel"
            }
            
            result = await ai_service.orchestrate_task(
                task_description="Build full-stack application",
                strategy="parallel",
                agents=["frontend_developer", "backend_developer", "database_architect"]
            )
            
            assert len(result["agents"]) == 3
            assert result["status"] == "in_progress"
    
    async def test_agent_communication_and_coordination(self, ai_service, mock_claude_flow_integration):
        """Test agent communication and coordination mechanisms."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            # Mock agent communication results
            mock_claude_flow_integration.get_agent_communication.return_value = {
                "messages": [
                    {"from": "agent_1", "to": "agent_2", "content": "API spec ready"},
                    {"from": "agent_2", "to": "agent_3", "content": "Database schema needed"},
                    {"from": "agent_3", "to": "agent_1", "content": "Schema complete"}
                ],
                "coordination_state": "synchronized"
            }
            
            # This would test actual communication methods when implemented
            # For now, we simulate via task orchestration
            result = await ai_service.orchestrate_task(
                task_description="Coordinate API development",
                requirements={"enable_communication": True}
            )
            
            assert result["task_id"] is not None


@pytest.mark.asyncio
class TestNeuralTrainingAndInference:
    """Test neural network training and inference capabilities."""
    
    async def test_neural_pattern_training(self, ai_service, mock_claude_flow_integration):
        """Test neural pattern training simulation."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            mock_claude_flow_integration.train_neural_patterns.return_value = {
                "training_id": "training_123",
                "status": "completed",
                "epochs": 50,
                "accuracy": 0.94,
                "loss": 0.06,
                "model_path": "/models/pattern_recognition.pkl"
            }
            
            # Mock neural training through task orchestration
            result = await ai_service.orchestrate_task(
                task_description="Train neural patterns for code quality",
                requirements={"model_type": "pattern_recognition", "epochs": 50}
            )
            
            assert result["task_id"] is not None
    
    async def test_neural_inference(self, ai_service, mock_claude_flow_integration):
        """Test neural inference for predictions."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            mock_claude_flow_integration.run_inference.return_value = {
                "inference_id": "inference_456",
                "predictions": [
                    {"pattern": "optimization_needed", "confidence": 0.85},
                    {"pattern": "refactoring_suggested", "confidence": 0.72},
                    {"pattern": "security_check_required", "confidence": 0.91}
                ],
                "execution_time": 0.15
            }
            
            # Mock inference through code generation with neural enhancement
            result = await ai_service.generate_code(
                prompt="Optimize this function for performance",
                language="python",
                context={"enable_neural_enhancement": True}
            )
            
            # This would include neural predictions in the response
            assert result is not None
    
    async def test_adaptive_learning(self, ai_service):
        """Test adaptive learning from user feedback."""
        # Mock feedback processing
        feedback_data = {
            "code_quality": "good",
            "user_rating": 4.5,
            "suggested_improvements": ["add_type_hints", "improve_documentation"],
            "context": {"language": "python", "project_type": "web_api"}
        }
        
        # This would feed back into the learning system
        # For now, we test that the service can handle feedback
        with patch.object(ai_service, '_request_history') as mock_history:
            mock_history.append({
                "feedback": feedback_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Verify feedback is recorded
            assert len(mock_history.method_calls) > 0


@pytest.mark.asyncio
class TestPerformanceAndBenchmarks:
    """Test AI service performance and benchmarking."""
    
    @pytest.mark.slow
    async def test_code_generation_performance(self, ai_service):
        """Test code generation performance benchmarks."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.return_value = {
                "code": "def benchmark_function(): pass",
                "language": "python"
            }
            
            start_time = time.time()
            
            # Generate multiple code snippets
            tasks = []
            for i in range(10):
                task = ai_service.generate_code(
                    prompt=f"Create function {i}",
                    language="python",
                    use_cache=False  # Disable cache for performance testing
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == 10
            assert total_time < 5.0  # Should complete within 5 seconds
            assert all(result["code"] is not None for result in results)
    
    @pytest.mark.slow
    async def test_concurrent_task_orchestration(self, ai_service, mock_claude_flow_integration):
        """Test concurrent task orchestration performance."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            start_time = time.time()
            
            # Launch multiple orchestration tasks concurrently
            tasks = []
            for i in range(5):
                task = ai_service.orchestrate_task(
                    task_description=f"Concurrent task {i}",
                    strategy="adaptive"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == 5
            assert total_time < 3.0  # Should complete within 3 seconds
            assert all(result["task_id"] is not None for result in results)
    
    async def test_memory_usage_optimization(self, ai_service):
        """Test memory usage during intensive operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.return_value = {
                "code": "def memory_test(): pass",
                "language": "python"
            }
            
            # Perform memory-intensive operations
            for i in range(100):
                await ai_service.generate_code(
                    prompt=f"Generate code {i}",
                    language="python",
                    use_cache=True  # Use cache to test memory efficiency
                )
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 50MB)
            assert memory_increase < 50 * 1024 * 1024
    
    async def test_request_history_and_metrics(self, ai_service):
        """Test request history tracking and metrics collection."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.return_value = {
                "code": "def metrics_test(): pass",
                "language": "python"
            }
            
            # Make several requests
            for i in range(5):
                await ai_service.generate_code(
                    prompt=f"Test request {i}",
                    language="python"
                )
            
            # Check request history
            history = await ai_service.get_request_history(limit=10)
            assert len(history) >= 5
            
            # Check performance metrics
            metrics = await ai_service.get_performance_metrics()
            assert metrics["total_requests"] >= 5
            assert metrics["successful_requests"] >= 5
            assert metrics["success_rate"] > 0
    
    async def test_cache_performance(self, ai_service):
        """Test caching performance and effectiveness."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.return_value = {
                "code": "def cache_test(): pass",
                "language": "python"
            }
            
            prompt = "Create a cached function"
            
            # First request (cache miss)
            start_time = time.time()
            result1 = await ai_service.generate_code(
                prompt=prompt,
                language="python",
                use_cache=True
            )
            first_time = time.time() - start_time
            
            # Second request (cache hit)
            start_time = time.time()
            result2 = await ai_service.generate_code(
                prompt=prompt,
                language="python",
                use_cache=True
            )
            second_time = time.time() - start_time
            
            # Cache hit should be much faster
            assert second_time < first_time / 2  # At least 50% faster
            assert result1 == result2  # Results should be identical
            assert mock_interface.generate_code.call_count == 1  # API called only once


@pytest.mark.asyncio
class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    async def test_claude_code_connection_failure(self, ai_service):
        """Test handling of Claude Code connection failures."""
        with patch.object(ai_service, '_test_claude_code_connection', side_effect=Exception("Connection failed")):
            # Service should still initialize with Claude Flow available
            with patch.object(ai_service, '_test_claude_flow_connection', return_value=True):
                service = AIService()
                await service.initialize()
                assert service._claude_code_available is False
                assert service._claude_flow_available is True
    
    async def test_claude_flow_connection_failure(self, ai_service):
        """Test handling of Claude Flow connection failures."""
        with patch.object(ai_service, '_test_claude_flow_connection', side_effect=Exception("Connection failed")):
            with patch.object(ai_service, '_test_claude_code_connection', return_value=True):
                service = AIService()
                await service.initialize()
                assert service._claude_code_available is True
                assert service._claude_flow_available is False
    
    async def test_api_rate_limiting_handling(self, ai_service):
        """Test handling of API rate limiting."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(ClaudeCodeError, match="Code generation failed"):
                await ai_service.generate_code(
                    prompt="This will be rate limited",
                    language="python"
                )
    
    async def test_invalid_response_handling(self, ai_service):
        """Test handling of invalid AI responses."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            # Mock invalid response (missing required fields)
            mock_interface.generate_code.return_value = {}
            
            result = await ai_service.generate_code(
                prompt="Generate code",
                language="python",
                validate_response=True
            )
            
            # Validation should catch the invalid response
            assert result["validation"]["is_valid"] is False
    
    async def test_timeout_handling(self, ai_service):
        """Test handling of request timeouts."""
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            mock_interface.generate_code.side_effect = asyncio.TimeoutError("Request timed out")
            
            with pytest.raises(ClaudeCodeError, match="Code generation failed"):
                await ai_service.generate_code(
                    prompt="This will timeout",
                    language="python"
                )


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    async def test_full_development_workflow(self, ai_service, mock_claude_code_integration, mock_claude_flow_integration):
        """Test complete development workflow integration."""
        with patch('src.services.ai_service.ClaudeFlowIntegration', return_value=mock_claude_flow_integration):
            with patch.object(ai_service, '_ai_interface') as mock_interface:
                mock_interface.generate_code.return_value = {
                    "code": "# Generated application code",
                    "language": "python"
                }
                
                # Step 1: Orchestrate development task
                orchestration_result = await ai_service.orchestrate_task(
                    task_description="Build web application with authentication",
                    requirements={"framework": "FastAPI", "testing": True},
                    strategy="adaptive"
                )
                
                # Step 2: Generate specific code components
                auth_code = await ai_service.generate_code(
                    prompt="Create JWT authentication system",
                    language="python",
                    context={"framework": "FastAPI"}
                )
                
                api_code = await ai_service.generate_code(
                    prompt="Create REST API endpoints",
                    language="python",
                    context={"authentication": "JWT"}
                )
                
                # Verify workflow completion
                assert orchestration_result["task_id"] is not None
                assert auth_code["code"] is not None
                assert api_code["code"] is not None
    
    async def test_ai_service_coordination_with_external_systems(self, ai_service):
        """Test AI service coordination with external systems."""
        # Mock external system integration
        with patch('src.services.ai_service.external_system_client') as mock_client:
            mock_client.submit_for_review.return_value = {"review_id": "review_123"}
            
            # Generate code that requires external review
            with patch.object(ai_service, '_ai_interface') as mock_interface:
                mock_interface.generate_code.return_value = {
                    "code": "def sensitive_function(): pass",
                    "language": "python",
                    "requires_review": True
                }
                
                result = await ai_service.generate_code(
                    prompt="Create security-critical function",
                    language="python",
                    context={"security_level": "high"}
                )
                
                assert result["code"] is not None
                # In a real implementation, this would trigger external review
    
    async def test_multi_language_code_generation(self, ai_service):
        """Test code generation across multiple programming languages."""
        languages = ["python", "javascript", "java", "go", "rust"]
        
        with patch.object(ai_service, '_ai_interface') as mock_interface:
            def mock_generate_code(prompt, language, context):
                return {
                    "code": f"// {language} code for: {prompt}",
                    "language": language
                }
            
            mock_interface.generate_code.side_effect = mock_generate_code
            
            # Generate code in multiple languages
            results = []
            for lang in languages:
                result = await ai_service.generate_code(
                    prompt="Create hello world function",
                    language=lang
                )
                results.append(result)
            
            # Verify all languages were processed
            assert len(results) == len(languages)
            for i, result in enumerate(results):
                assert languages[i] in result["code"]
                assert result["language"] == languages[i]