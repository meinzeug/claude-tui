"""
Comprehensive unit tests for AI Service.

Tests cover:
- Service initialization and health checks
- Code generation with validation
- Task orchestration via Claude Flow
- Response validation and caching
- Error handling and fallback mechanisms
- Performance monitoring
- Edge cases and anti-hallucination measures
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

from services.ai_service import AIService
from core.exceptions import AIServiceError, ClaudeCodeError, ClaudeFlowError, ValidationError


class TestAIServiceInitialization:
    """Test AI Service initialization and setup."""
    
    @pytest.mark.unit
    async def test_service_initialization_success(self):
        """Test successful service initialization."""
        service = AIService()
        
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    # Mock successful connections
                    mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                    mock_claude_flow.return_value.test_connection.return_value = {'status': 'connected'}
                    
                    await service.initialize()
                    
                    assert service._initialized is True
                    assert service._claude_code_available is True
                    assert service._claude_flow_available is True
                    assert service._ai_interface is not None
    
    @pytest.mark.unit
    async def test_service_initialization_partial_providers(self):
        """Test initialization with only one provider available."""
        service = AIService()
        
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    # Mock Claude Code available, Claude Flow not
                    mock_claude_code.return_value.test_connection.return_value = {'status': 'connected'}
                    mock_claude_flow.return_value.test_connection.side_effect = Exception("Connection failed")
                    
                    await service.initialize()
                    
                    assert service._initialized is True
                    assert service._claude_code_available is True
                    assert service._claude_flow_available is False
    
    @pytest.mark.unit
    async def test_service_initialization_no_providers(self):
        """Test initialization fails when no providers available."""
        service = AIService()
        
        with patch('services.ai_service.AIInterface') as mock_ai_interface:
            with patch('services.ai_service.ClaudeCodeIntegration') as mock_claude_code:
                with patch('services.ai_service.ClaudeFlowIntegration') as mock_claude_flow:
                    # Mock both providers unavailable
                    mock_claude_code.return_value.test_connection.side_effect = Exception("Connection failed")
                    mock_claude_flow.return_value.test_connection.side_effect = Exception("Connection failed")
                    
                    with pytest.raises(AIServiceError) as excinfo:
                        await service.initialize()
                    
                    assert "No AI providers available" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_health_check(self, ai_service):
        """Test AI service health check."""
        health = await ai_service.health_check()
        
        assert isinstance(health, dict)
        assert health['service'] == 'AIService'
        assert health['status'] == 'healthy'
        assert 'claude_code_available' in health
        assert 'claude_flow_available' in health
        assert 'response_cache_size' in health
        assert 'request_history_size' in health


class TestAIServiceCodeGeneration:
    """Test AI Service code generation functionality."""
    
    @pytest.mark.unit
    async def test_generate_code_success(self, ai_service, sample_python_code):
        """Test successful code generation."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': sample_python_code,
                'language': 'python',
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'model': 'claude-3'
                }
            }
            
            result = await ai_service.generate_code(
                prompt="Create a fibonacci function",
                language="python",
                validate_response=True
            )
            
            assert result['code'] == sample_python_code
            assert result['language'] == 'python'
            assert 'validation' in result
            assert 'metadata' in result
            
            mock_generate.assert_called_once()
    
    @pytest.mark.unit
    async def test_generate_code_with_validation_failure(self, ai_service):
        """Test code generation with validation failure."""
        invalid_code = "def broken_function(\n    return 'invalid'"
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': invalid_code,
                'language': 'python'
            }
            
            result = await ai_service.generate_code(
                prompt="Create a function",
                language="python",
                validate_response=True
            )
            
            assert result['code'] == invalid_code
            assert 'validation' in result
            assert result['validation']['is_valid'] is False
            assert len(result['validation']['errors']) > 0
    
    @pytest.mark.unit
    async def test_generate_code_with_caching(self, ai_service, sample_python_code):
        """Test code generation with response caching."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': sample_python_code,
                'language': 'python'
            }
            
            # First call
            result1 = await ai_service.generate_code(
                prompt="Create a fibonacci function",
                language="python",
                use_cache=True
            )
            
            # Second call with same parameters (should use cache)
            result2 = await ai_service.generate_code(
                prompt="Create a fibonacci function",
                language="python",
                use_cache=True
            )
            
            assert result1 == result2
            # Should only call the AI interface once
            mock_generate.assert_called_once()
    
    @pytest.mark.unit
    async def test_generate_code_claude_code_unavailable(self, ai_service):
        """Test code generation when Claude Code is unavailable."""
        ai_service._claude_code_available = False
        
        with pytest.raises(ClaudeCodeError) as excinfo:
            await ai_service.generate_code("Create a function")
        
        assert "Claude Code not available" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_generate_code_with_context(self, ai_service, sample_python_code):
        """Test code generation with context."""
        context = {
            'project_type': 'web_api',
            'framework': 'fastapi',
            'existing_functions': ['authenticate_user', 'validate_input']
        }
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': sample_python_code,
                'language': 'python'
            }
            
            result = await ai_service.generate_code(
                prompt="Create user management function",
                language="python",
                context=context
            )
            
            mock_generate.assert_called_once_with(
                prompt="Create user management function",
                language="python",
                context=context
            )
            assert result['code'] == sample_python_code


class TestAIServiceTaskOrchestration:
    """Test AI Service task orchestration functionality."""
    
    @pytest.mark.unit
    async def test_orchestrate_task_success(self, ai_service):
        """Test successful task orchestration."""
        with patch('services.ai_service.ClaudeFlowIntegration') as mock_flow:
            mock_flow_instance = AsyncMock()
            mock_flow.return_value = mock_flow_instance
            
            mock_flow_instance.orchestrate_task.return_value = {
                'task_id': 'task-123',
                'status': 'running',
                'agents': ['coder', 'reviewer'],
                'estimated_completion': '2025-01-01T01:00:00Z'
            }
            
            result = await ai_service.orchestrate_task(
                task_description="Build a REST API",
                requirements={'framework': 'fastapi', 'database': 'postgresql'},
                strategy='adaptive'
            )
            
            assert result['task_id'] == 'task-123'
            assert result['status'] == 'running'
            assert len(result['agents']) == 2
            
            mock_flow_instance.orchestrate_task.assert_called_once()
    
    @pytest.mark.unit
    async def test_orchestrate_task_claude_flow_unavailable(self, ai_service):
        """Test task orchestration when Claude Flow is unavailable."""
        ai_service._claude_flow_available = False
        
        with pytest.raises(ClaudeFlowError) as excinfo:
            await ai_service.orchestrate_task("Build an API")
        
        assert "Claude Flow not available" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_orchestrate_task_with_agents(self, ai_service):
        """Test task orchestration with specific agents."""
        with patch('services.ai_service.ClaudeFlowIntegration') as mock_flow:
            mock_flow_instance = AsyncMock()
            mock_flow.return_value = mock_flow_instance
            
            mock_flow_instance.orchestrate_task.return_value = {
                'task_id': 'task-456',
                'agents': ['coder', 'tester'],
                'strategy': 'parallel'
            }
            
            result = await ai_service.orchestrate_task(
                task_description="Create user authentication",
                agents=['coder', 'tester'],
                strategy='parallel'
            )
            
            assert result['task_id'] == 'task-456'
            assert 'coder' in result['agents']
            assert 'tester' in result['agents']


class TestAIServiceResponseValidation:
    """Test AI Service response validation functionality."""
    
    @pytest.mark.unit
    async def test_validate_code_response_valid(self, ai_service, sample_python_code):
        """Test validation of valid code response."""
        response = {
            'code': sample_python_code,
            'language': 'python'
        }
        
        result = await ai_service.validate_response(
            response=response,
            validation_type='code',
            criteria={'language': 'python'}
        )
        
        assert result['is_valid'] is True
        assert result['score'] > 0.8
        assert len(result['errors']) == 0
    
    @pytest.mark.unit
    async def test_validate_code_response_invalid(self, ai_service):
        """Test validation of invalid code response."""
        response = {
            'code': 'def broken(\n    return invalid',
            'language': 'python'
        }
        
        result = await ai_service.validate_response(
            response=response,
            validation_type='code',
            criteria={'language': 'python'}
        )
        
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert len(result['errors']) > 0
    
    @pytest.mark.unit
    async def test_validate_text_response(self, ai_service):
        """Test validation of text response."""
        response = {
            'content': 'This is a comprehensive explanation of the problem and solution approach.'
        }
        
        result = await ai_service.validate_response(
            response=response,
            validation_type='text',
            criteria={'min_length': 10}
        )
        
        assert result['is_valid'] is True
        assert result['score'] > 0.0
    
    @pytest.mark.unit
    async def test_validate_general_response_missing_fields(self, ai_service):
        """Test validation of general response with missing required fields."""
        response = {
            'partial_data': 'some value'
        }
        
        result = await ai_service.validate_response(
            response=response,
            validation_type='general',
            criteria={'required_fields': ['name', 'description', 'status']}
        )
        
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert any('Missing required fields' in error for error in result['errors'])


class TestAIServicePlaceholderDetection:
    """Test AI Service placeholder detection and anti-hallucination."""
    
    @pytest.mark.unit
    async def test_detect_python_placeholders(self, ai_service):
        """Test detection of Python placeholder patterns."""
        code_with_placeholders = '''
def incomplete_function():
    # TODO: Implement this function
    pass  # implement later

class EmptyClass:
    pass

def another_function():
    ...  # placeholder
    
def broken():
    raise NotImplementedError("Fix this")
'''
        
        response = {'code': code_with_placeholders}
        result = await ai_service.validate_response(
            response=response,
            validation_type='code',
            criteria={'language': 'python'}
        )
        
        assert len(result['warnings']) > 0
        assert any('placeholder' in warning.lower() for warning in result['warnings'])
        assert result['score'] < 1.0  # Score should be reduced due to placeholders
    
    @pytest.mark.unit
    async def test_detect_javascript_placeholders(self, ai_service):
        """Test detection of JavaScript placeholder patterns."""
        code_with_placeholders = '''
function incompleteFunction() {
    // TODO: Implement this
    throw new Error("Not implemented");
}

const anotherFunction = () => {
    // FIXME: Complete implementation
    console.log("TODO: Add logic here");
};
'''
        
        response = {'code': code_with_placeholders}
        result = await ai_service.validate_response(
            response=response,
            validation_type='code',
            criteria={'language': 'javascript'}
        )
        
        assert len(result['warnings']) > 0
        assert result['score'] < 1.0


class TestAIServiceErrorHandling:
    """Test AI Service error handling and resilience."""
    
    @pytest.mark.unit
    async def test_ai_interface_failure_handling(self, ai_service):
        """Test handling of AI interface failures."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.side_effect = Exception("AI service temporarily unavailable")
            
            with pytest.raises(ClaudeCodeError) as excinfo:
                await ai_service.generate_code("Create a function")
            
            assert "Code generation failed" in str(excinfo.value)
            assert len(ai_service._request_history) > 0
            assert ai_service._request_history[-1]['response']['success'] is False
    
    @pytest.mark.unit
    async def test_timeout_handling(self, ai_service):
        """Test handling of operation timeouts."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than typical timeout
                return {'code': 'def test(): pass'}
            
            mock_generate.side_effect = slow_response
            
            with pytest.raises(ClaudeCodeError):
                await ai_service.generate_code("Create a function")
    
    @pytest.mark.unit
    async def test_invalid_input_handling(self, ai_service):
        """Test handling of invalid input parameters."""
        with pytest.raises(ValidationError):
            await ai_service.generate_code("")  # Empty prompt
        
        with pytest.raises(ValidationError):
            await ai_service.generate_code(None)  # None prompt
    
    @pytest.mark.edge_case
    async def test_memory_pressure_handling(self, ai_service):
        """Test behavior under memory pressure with large requests."""
        # Simulate large context
        large_context = {
            'large_data': 'x' * (10 * 1024 * 1024),  # 10MB of data
            'metadata': {'size': 'very_large'}
        }
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            result = await ai_service.generate_code(
                prompt="Create a function",
                context=large_context
            )
            
            assert result['code'] == 'def test(): pass'
            mock_generate.assert_called_once()


class TestAIServicePerformanceMonitoring:
    """Test AI Service performance monitoring and metrics."""
    
    @pytest.mark.unit
    async def test_request_history_tracking(self, ai_service, sample_python_code):
        """Test that request history is properly tracked."""
        initial_history_size = len(ai_service._request_history)
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {
                'code': sample_python_code,
                'language': 'python'
            }
            
            await ai_service.generate_code("Create a function")
            
            assert len(ai_service._request_history) == initial_history_size + 1
            
            latest_request = ai_service._request_history[-1]
            assert latest_request['request']['type'] == 'code_generation'
            assert latest_request['response']['success'] is True
    
    @pytest.mark.unit
    async def test_performance_metrics(self, ai_service):
        """Test performance metrics collection."""
        # Generate some requests to have metrics
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            await ai_service.generate_code("Test 1")
            await ai_service.generate_code("Test 2")
            
            # Simulate one failure
            mock_generate.side_effect = Exception("Test error")
            try:
                await ai_service.generate_code("Test 3")
            except:
                pass
        
        metrics = await ai_service.get_performance_metrics()
        
        assert metrics['total_requests'] >= 3
        assert metrics['successful_requests'] >= 2
        assert 0 <= metrics['success_rate'] <= 1
        assert 'cache_size' in metrics
        assert 'providers_count' in metrics
    
    @pytest.mark.unit
    async def test_cache_management(self, ai_service):
        """Test response cache management."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            # Add item to cache
            await ai_service.generate_code("Test prompt", use_cache=True)
            
            assert len(ai_service._response_cache) > 0
            
            # Clear cache
            await ai_service.clear_cache()
            
            assert len(ai_service._response_cache) == 0
    
    @pytest.mark.unit
    async def test_request_history_filtering(self, ai_service):
        """Test request history filtering functionality."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            await ai_service.generate_code("Test")
        
        with patch('services.ai_service.ClaudeFlowIntegration') as mock_flow:
            mock_flow_instance = AsyncMock()
            mock_flow.return_value = mock_flow_instance
            mock_flow_instance.orchestrate_task.return_value = {'task_id': 'test'}
            
            await ai_service.orchestrate_task("Test task")
        
        # Test filtering
        code_history = await ai_service.get_request_history(
            operation_filter='code_generation'
        )
        
        assert len(code_history) >= 1
        assert all(req['request']['type'] == 'code_generation' for req in code_history)
        
        # Test limit
        limited_history = await ai_service.get_request_history(limit=1)
        assert len(limited_history) <= 1


class TestAIServiceEdgeCases:
    """Test AI Service edge cases and boundary conditions."""
    
    @pytest.mark.edge_case
    async def test_empty_code_response(self, ai_service):
        """Test handling of empty code response."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': '', 'language': 'python'}
            
            result = await ai_service.validate_response(
                response={'code': ''},
                validation_type='code'
            )
            
            assert result['is_valid'] is False
            assert any('No code content found' in error for error in result['errors'])
    
    @pytest.mark.edge_case
    async def test_very_long_code_response(self, ai_service):
        """Test handling of very long code response."""
        very_long_code = 'def test():\n    pass\n' * 10000  # ~200KB of code
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': very_long_code, 'language': 'python'}
            
            result = await ai_service.generate_code("Create a function")
            
            assert result['code'] == very_long_code
            assert len(result['code']) > 100000
    
    @pytest.mark.edge_case
    async def test_unicode_content_handling(self, ai_service):
        """Test handling of Unicode content in responses."""
        unicode_code = '''
def greet():
    return "Hello 世界! 🌍 Здравствуй мир!"
    
class UnicodeTest:
    """Testing with émojis and spëcial characters."""
    pass
'''
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': unicode_code, 'language': 'python'}
            
            result = await ai_service.generate_code("Create greeting function")
            
            assert result['code'] == unicode_code
            assert '世界' in result['code']
            assert '🌍' in result['code']
    
    @pytest.mark.edge_case
    async def test_malformed_response_handling(self, ai_service):
        """Test handling of malformed AI responses."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            # Return malformed response missing required fields
            mock_generate.return_value = {'incomplete': 'response'}
            
            result = await ai_service.generate_code("Create a function")
            
            # Service should handle gracefully
            assert isinstance(result, dict)
    
    @pytest.mark.edge_case
    async def test_concurrent_requests(self, ai_service):
        """Test handling of concurrent requests."""
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            # Create multiple concurrent requests
            tasks = [
                ai_service.generate_code(f"Create function {i}")
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all(result['code'] == 'def test(): pass' for result in results)
            
            # Check that all requests were tracked
            assert len(ai_service._request_history) >= 10
    
    @pytest.mark.performance
    async def test_performance_under_load(self, ai_service, performance_test_config):
        """Test AI service performance under load."""
        from tests.conftest import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        with patch.object(ai_service._ai_interface, 'generate_code') as mock_generate:
            mock_generate.return_value = {'code': 'def test(): pass'}
            
            monitor.start()
            
            # Execute multiple operations
            tasks = []
            for i in range(performance_test_config['concurrent_operations']):
                task = ai_service.generate_code(f"Create function {i}")
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            monitor.stop()
            
            # Assert performance meets requirements
            monitor.assert_performance(performance_test_config['max_execution_time'])
            
            # Verify all operations completed successfully
            assert len(tasks) == performance_test_config['concurrent_operations']