"""
Integration Tests for ClaudeCodeDirectClient

Tests the integration of ClaudeCodeDirectClient with the existing
claude-tui system components and workflows.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from claude_tui.integrations.claude_code_direct_client import ClaudeCodeDirectClient
from claude_tui.core.config_manager import ConfigManager
from claude_tui.models.project import Project
from claude_tui.models.ai_models import ReviewCriteria, CodeResult


class TestClaudeCodeDirectIntegration:
    """Test integration with existing claude-tui components."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create temporary .cc file
        self.temp_cc_fd, self.temp_cc_file = tempfile.mkstemp(suffix='.cc')
        with os.fdopen(self.temp_cc_fd, 'w') as f:
            f.write('sk-ant-oat01-integration-test-token')
        
        # Create temporary project directory
        self.temp_project_dir = tempfile.mkdtemp()
        
        # Mock CLI validation to avoid requiring actual Claude CLI
        with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
            self.client = ClaudeCodeDirectClient(
                oauth_token_file=self.temp_cc_file,
                working_directory=self.temp_project_dir
            )
    
    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.temp_cc_file):
            os.unlink(self.temp_cc_file)
        
        if os.path.exists(self.temp_project_dir):
            import shutil
            shutil.rmtree(self.temp_project_dir)
        
        self.client.cleanup_session()
    
    def test_config_manager_integration(self):
        """Test integration with ConfigManager."""
        config_manager = ConfigManager()
        
        # Test factory method with config manager
        with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
            client = ClaudeCodeDirectClient.create_from_config(
                config_manager=config_manager,
                claude_code_path="test-claude"
            )
        
        assert client.config_manager == config_manager
        assert client.claude_code_path == "test-claude"
        
        client.cleanup_session()
    
    def test_project_integration(self):
        """Test integration with Project model."""
        # Create a mock project
        project = Mock(spec=Project)
        project.path = Path(self.temp_project_dir)
        project.name = "test-project"
        
        # Create some test files in the project directory
        test_file = Path(self.temp_project_dir) / "main.py"
        test_file.write_text('def hello(): print("Hello, World!")')
        
        # Test that client can work with project context
        session_info = self.client.get_session_info()
        assert session_info['working_directory'] == self.temp_project_dir
    
    @pytest.mark.asyncio
    async def test_code_review_integration(self):
        """Test integration with existing code review workflow."""
        # Mock code review criteria
        criteria = Mock(spec=ReviewCriteria)
        criteria.focus_areas = ["syntax", "style", "performance"]
        criteria.strictness = "medium"
        
        # Mock project
        project = Mock(spec=Project)
        project.path = Path(self.temp_project_dir)
        
        # Sample code to review
        code_to_review = '''
def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
'''
        
        # Mock successful CLI execution
        mock_subprocess_result = Mock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = '''
Code Review Results:

Good points:
• Function has clear logic flow
• Handles base case correctly

Issues:
• Missing type hints
• No docstring provided
• Could use more descriptive variable names

Suggestions:
• Add type annotations for parameters and return value
• Include comprehensive docstring
• Consider using more descriptive variable names
'''
        mock_subprocess_result.stderr = ''
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_subprocess_result):
            review_result = await self.client.review_code(
                code=code_to_review,
                criteria=criteria,
                project=project
            )
        
        # Verify review result structure
        assert hasattr(review_result, 'overall_score')
        assert hasattr(review_result, 'issues')
        assert hasattr(review_result, 'suggestions')
        assert hasattr(review_result, 'compliments')
        assert hasattr(review_result, 'summary')
        
        # Verify parsed content
        assert len(review_result.issues) > 0
        assert len(review_result.suggestions) > 0
        assert len(review_result.compliments) > 0
        
        # Check that issues contain expected content
        issues_text = ' '.join([issue.get('description', str(issue)) for issue in review_result.issues])
        assert 'type hints' in issues_text.lower()
    
    @pytest.mark.asyncio
    async def test_task_execution_with_context(self):
        """Test task execution with project context."""
        # Mock successful task execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '''
Here's the requested function:

```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using an iterative approach.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative numbers")
    elif n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
```

This implementation is efficient with O(n) time complexity and O(1) space complexity.
'''
        mock_result.stderr = ''
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.execute_task_via_cli(
                task_description="Write a Python function to calculate Fibonacci numbers with proper error handling and documentation",
                context={
                    "language": "python",
                    "style": "clean and documented",
                    "project_path": self.temp_project_dir,
                    "include_error_handling": True,
                    "include_type_hints": True
                },
                working_directory=self.temp_project_dir,
                timeout=120
            )
        
        # Verify result structure
        assert result['success'] is True
        assert 'execution_id' in result
        assert 'execution_time' in result
        assert 'generated_code' in result
        
        # Verify generated code contains expected elements
        generated_code = result['generated_code']
        assert 'def fibonacci' in generated_code
        assert 'int' in generated_code  # Type hints
        assert '"""' in generated_code  # Docstring
        assert 'ValueError' in generated_code  # Error handling
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration with system error handling."""
        # Mock CLI execution failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ''
        mock_result.stderr = 'Error: Invalid task description provided'
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            result = await self.client.execute_task_via_cli(
                task_description="Invalid or unclear task",
                timeout=30
            )
        
        # Verify error handling
        assert result['success'] is False
        assert 'error' in result
        assert 'Invalid task description' in result['error']
        assert 'execution_id' in result
        assert 'execution_time' in result
    
    @pytest.mark.asyncio
    async def test_validation_workflow_integration(self):
        """Test integration with existing validation workflows."""
        # Sample code with potential issues
        code_with_issues = '''
def process_data(data):
    # Missing type hints and error handling
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''
        
        # Mock validation result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '''
Validation Results:

Issues:
• Missing type hints for function parameters and return value
• No error handling for invalid input
• Function lacks docstring

Suggestions:
• Add type annotations: def process_data(data: List[float]) -> List[float]
• Add input validation and error handling
• Include comprehensive docstring
• Consider edge cases (empty list, None values)
'''
        mock_result.stderr = ''
        
        with patch.object(self.client, '_execute_cli_command', return_value=mock_result):
            validation_result = await self.client.validate_code_via_cli(
                code=code_with_issues,
                validation_rules=[
                    "check type hints",
                    "verify error handling",
                    "ensure documentation",
                    "review best practices"
                ],
                context={
                    "language": "python",
                    "strictness": "high",
                    "project_context": "data processing module"
                }
            )
        
        # Verify validation result structure
        assert validation_result['valid'] is True  # CLI succeeded, but code has issues
        assert 'issues' in validation_result
        assert 'suggestions' in validation_result
        assert len(validation_result['issues']) > 0
        assert len(validation_result['suggestions']) > 0
        
        # Verify specific issues were identified
        all_issues = ' '.join(validation_result['issues'])
        assert 'type hints' in all_issues.lower()
        assert 'error handling' in all_issues.lower()
    
    def test_session_management_integration(self):
        """Test session management with system monitoring."""
        # Get initial session info
        session_info = self.client.get_session_info()
        initial_execution_count = session_info['execution_count']
        
        # Simulate some executions by incrementing counter
        self.client._execution_count += 3
        
        # Get updated session info
        updated_session_info = self.client.get_session_info()
        
        # Verify session tracking
        assert updated_session_info['execution_count'] == initial_execution_count + 3
        assert updated_session_info['session_id'] == session_info['session_id']
        assert updated_session_info['oauth_token_available'] is True
        
        # Test session cleanup
        session_id_before_cleanup = self.client.session_id
        self.client.cleanup_session()
        
        # Verify cleanup effects
        assert self.client.oauth_token is None  # Token cleared for security
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations handling."""
        # Mock CLI executions
        mock_result1 = Mock()
        mock_result1.returncode = 0
        mock_result1.stdout = 'Task 1 completed\n```python\ndef task1(): pass\n```'
        mock_result1.stderr = ''
        
        mock_result2 = Mock()
        mock_result2.returncode = 0
        mock_result2.stdout = 'Task 2 completed\n```python\ndef task2(): pass\n```'
        mock_result2.stderr = ''
        
        # Create a side effect that returns different results for different calls
        def mock_execute_side_effect(cmd, timeout, working_directory=None, execution_id=None):
            if 'task 1' in ' '.join(cmd).lower():
                return mock_result1
            else:
                return mock_result2
        
        with patch.object(self.client, '_execute_cli_command', side_effect=mock_execute_side_effect):
            # Execute multiple tasks concurrently
            task1 = self.client.execute_task_via_cli("Execute task 1", timeout=30)
            task2 = self.client.execute_task_via_cli("Execute task 2", timeout=30)
            
            # Wait for both to complete
            results = await asyncio.gather(task1, task2)
        
        # Verify both tasks completed successfully
        assert len(results) == 2
        assert all(result['success'] for result in results)
        assert results[0]['execution_id'] != results[1]['execution_id']  # Different execution IDs
        
        # Verify execution count was properly tracked
        final_session_info = self.client.get_session_info()
        assert final_session_info['execution_count'] >= 2


class TestCompatibilityWithExistingCode:
    """Test backward compatibility with existing code using ClaudeCodeClient."""
    
    @pytest.mark.asyncio
    async def test_drop_in_replacement_compatibility(self):
        """Test that ClaudeCodeDirectClient can serve as drop-in replacement."""
        # Create temporary token file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.cc')
        with os.fdopen(temp_fd, 'w') as f:
            f.write('sk-ant-oat01-compatibility-test')
        
        try:
            with patch.object(ClaudeCodeDirectClient, '_validate_claude_code_cli'):
                client = ClaudeCodeDirectClient(oauth_token_file=temp_file)
            
            # Test that it has the expected interface methods
            assert hasattr(client, 'execute_task_via_cli')
            assert hasattr(client, 'validate_code_via_cli')
            assert hasattr(client, 'refactor_code_via_cli')
            assert hasattr(client, 'review_code')  # Legacy compatibility
            assert hasattr(client, 'health_check')
            assert hasattr(client, 'cleanup_session')
            
            # Test session management interface
            assert hasattr(client, 'get_session_info')
            assert hasattr(client, 'session_id')
            
            client.cleanup_session()
        
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    # Run the integration tests
    pytest.main([__file__, "-v"])