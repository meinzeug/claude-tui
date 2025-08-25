"""
Unit tests for AI Interface functionality.

Tests Claude Code integration, validation, and AI-powered operations.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess
import json
import asyncio
from typing import Dict, Any, List


class TestAIInterface:
    """Test suite for AI Interface class."""
    
    @pytest.fixture
    def ai_interface(self):
        """Create AI interface instance."""
        # This will be: from claude_tiu.core.ai_interface import ClaudeInterface
        # For now, create a mock implementation
        
        class MockAIInterface:
            def __init__(self, api_key="test-key", timeout=30):
                self.api_key = api_key
                self.timeout = timeout
                self.command_history = []
            
            def execute_claude_code(self, prompt, **kwargs):
                """Execute Claude Code command."""
                cmd = ["claude-code", "--prompt", prompt]
                if "output" in kwargs:
                    cmd.extend(["--output", kwargs["output"]])
                
                self.command_history.append(cmd)
                
                # Mock subprocess execution
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return {"status": "success", "output": result.stdout}
                else:
                    raise RuntimeError(f"Claude Code execution failed: {result.stderr}")
            
            async def execute_claude_code_async(self, prompt, **kwargs):
                """Async version of Claude Code execution."""
                return await asyncio.to_thread(self.execute_claude_code, prompt, **kwargs)
            
            def validate_project(self, project_data):
                """Validate project configuration."""
                required_fields = ["name", "template"]
                return all(field in project_data and project_data[field] for field in required_fields)
            
            async def validate_code(self, code):
                """Validate code authenticity."""
                # Simulate AI-based code validation
                placeholder_indicators = [
                    "TODO", "FIXME", "NotImplementedError", "pass", "console.log"
                ]
                
                has_placeholders = any(indicator in code for indicator in placeholder_indicators)
                
                return {
                    "authentic": not has_placeholders,
                    "confidence": 0.9 if not has_placeholders else 0.3,
                    "issues": ["Contains placeholders"] if has_placeholders else [],
                    "suggestions": ["Implement TODO items"] if has_placeholders else []
                }
        
        return MockAIInterface()
    
    def test_ai_interface_initialization(self, ai_interface):
        """Test AI interface initialization."""
        # Assert
        assert ai_interface.api_key == "test-key"
        assert ai_interface.timeout == 30
        assert ai_interface.command_history == []
    
    @patch('subprocess.run')
    def test_execute_claude_code_success(self, mock_run, ai_interface):
        """Test successful Claude Code execution."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='{"result": "success", "output": "Code generated"}',
            stderr='',
            returncode=0
        )
        
        # Act
        result = ai_interface.execute_claude_code("Generate a Python function")
        
        # Assert
        assert result["result"] == "success"
        assert result["output"] == "Code generated"
        mock_run.assert_called_once()
        
        # Verify command construction
        called_args = mock_run.call_args[0][0]
        assert "claude-code" in called_args
        assert "--prompt" in called_args
        assert "Generate a Python function" in called_args
    
    @patch('subprocess.run')
    def test_execute_claude_code_with_output_file(self, mock_run, ai_interface):
        """Test Claude Code execution with output file."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='{"status": "success"}',
            stderr='',
            returncode=0
        )
        
        # Act
        result = ai_interface.execute_claude_code(
            "Generate code",
            output="/tmp/output.py"
        )
        
        # Assert
        assert result["status"] == "success"
        
        # Verify output parameter was passed
        called_args = mock_run.call_args[0][0]
        assert "--output" in called_args
        assert "/tmp/output.py" in called_args
    
    @patch('subprocess.run')
    def test_execute_claude_code_non_json_response(self, mock_run, ai_interface):
        """Test Claude Code execution with non-JSON response."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='Plain text response',
            stderr='',
            returncode=0
        )
        
        # Act
        result = ai_interface.execute_claude_code("Generate function")
        
        # Assert
        assert result["status"] == "success"
        assert result["output"] == "Plain text response"
    
    @patch('subprocess.run')
    def test_execute_claude_code_failure(self, mock_run, ai_interface):
        """Test Claude Code execution with error."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='',
            stderr='API Error: Rate limit exceeded',
            returncode=1
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Claude Code execution failed"):
            ai_interface.execute_claude_code("Generate function")
    
    @patch('subprocess.run')
    def test_execute_claude_code_timeout(self, mock_run, ai_interface):
        """Test Claude Code execution timeout."""
        # Arrange
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["claude-code"], timeout=30
        )
        
        # Act & Assert
        with pytest.raises(subprocess.TimeoutExpired):
            ai_interface.execute_claude_code("Long running task")
    
    @pytest.mark.asyncio
    async def test_execute_claude_code_async(self, ai_interface):
        """Test asynchronous Claude Code execution."""
        # Arrange - Mock the sync method
        ai_interface.execute_claude_code = Mock(return_value={
            "status": "success",
            "output": "Async execution complete"
        })
        
        # Act
        result = await ai_interface.execute_claude_code_async("Generate async code")
        
        # Assert
        assert result["status"] == "success"
        assert result["output"] == "Async execution complete"
    
    def test_validate_project_valid_data(self, ai_interface):
        """Test project validation with valid data."""
        # Arrange
        valid_project = {
            "name": "test-project",
            "template": "python",
            "description": "A test project"
        }
        
        # Act
        result = ai_interface.validate_project(valid_project)
        
        # Assert
        assert result is True
    
    @pytest.mark.parametrize("invalid_project", [
        {},  # Empty
        {"name": ""},  # Empty name
        {"template": "python"},  # Missing name
        {"name": "test"},  # Missing template
        {"name": None, "template": "python"},  # None name
        {"name": "test", "template": ""},  # Empty template
    ])
    def test_validate_project_invalid_data(self, ai_interface, invalid_project):
        """Test project validation with invalid data."""
        # Act
        result = ai_interface.validate_project(invalid_project)
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_code_authentic(self, ai_interface):
        """Test code validation for authentic code."""
        # Arrange
        authentic_code = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b
'''
        
        # Act
        result = await ai_interface.validate_code(authentic_code)
        
        # Assert
        assert result["authentic"] is True
        assert result["confidence"] > 0.8
        assert len(result["issues"]) == 0
        assert len(result["suggestions"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_code_with_placeholders(self, ai_interface):
        """Test code validation for code with placeholders."""
        # Arrange
        placeholder_code = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    # TODO: implement validation
    # TODO: add error handling
    pass
'''
        
        # Act
        result = await ai_interface.validate_code(placeholder_code)
        
        # Assert
        assert result["authentic"] is False
        assert result["confidence"] < 0.5
        assert "Contains placeholders" in result["issues"]
        assert "Implement TODO items" in result["suggestions"]
    
    @pytest.mark.asyncio
    async def test_validate_code_javascript_placeholders(self, ai_interface):
        """Test code validation detecting JavaScript-like placeholders."""
        # Arrange
        js_placeholder_code = '''
def debug_function():
    console.log("Debug info")
    return "result"
'''
        
        # Act
        result = await ai_interface.validate_code(js_placeholder_code)
        
        # Assert
        assert result["authentic"] is False
        assert result["confidence"] < 0.5
    
    def test_command_history_tracking(self, ai_interface):
        """Test that command history is tracked."""
        # Arrange
        ai_interface.execute_claude_code = Mock(return_value={"status": "success"})
        
        # Act
        ai_interface.execute_claude_code("First command")
        ai_interface.execute_claude_code("Second command", output="test.py")
        
        # Assert
        assert len(ai_interface.command_history) == 2
        
        first_cmd = ai_interface.command_history[0]
        assert "First command" in first_cmd
        
        second_cmd = ai_interface.command_history[1]
        assert "Second command" in second_cmd
        assert "test.py" in second_cmd
    
    @pytest.mark.parametrize("prompt,expected_in_cmd", [
        ("Simple prompt", "Simple prompt"),
        ("Multi\\nline\\nprompt", "Multi\\nline\\nprompt"),
        ("Prompt with 'quotes'", "Prompt with 'quotes'"),
        ("Prompt with special chars: @#$%", "Prompt with special chars: @#$%"),
    ])
    def test_prompt_handling(self, ai_interface, prompt, expected_in_cmd):
        """Test various prompt formats are handled correctly."""
        # Arrange
        ai_interface.execute_claude_code = Mock(return_value={"status": "success"})
        
        # Act
        ai_interface.execute_claude_code(prompt)
        
        # Assert
        last_cmd = ai_interface.command_history[-1]
        assert expected_in_cmd in str(last_cmd)
    
    def test_timeout_configuration(self):
        """Test timeout configuration."""
        # Arrange & Act
        interface1 = MockAIInterface(timeout=60)
        interface2 = MockAIInterface(timeout=10)
        
        # Assert
        assert interface1.timeout == 60
        assert interface2.timeout == 10
    
    @pytest.mark.integration
    @patch('subprocess.run')
    def test_integration_with_real_command_structure(self, mock_run):
        """Test integration with realistic Claude Code command structure."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                "status": "success",
                "files_created": ["main.py", "tests/test_main.py"],
                "analysis": {
                    "functions_created": 3,
                    "tests_created": 5,
                    "coverage": 0.95
                }
            }),
            stderr='',
            returncode=0
        )
        
        interface = MockAIInterface()
        
        # Act
        result = interface.execute_claude_code(
            "Create a calculator with comprehensive tests",
            output="calculator.py"
        )
        
        # Assert
        assert result["status"] == "success"
        assert "files_created" in result
        assert "analysis" in result
        assert result["analysis"]["coverage"] == 0.95
        
        # Verify realistic command structure
        called_args = mock_run.call_args[0][0]
        expected_structure = [
            "claude-code",
            "--prompt",
            "Create a calculator with comprehensive tests", 
            "--output",
            "calculator.py"
        ]
        assert called_args == expected_structure


class TestAIInterfaceErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def ai_interface(self):
        """Create AI interface for error testing."""
        class MockAIInterface:
            def __init__(self):
                self.command_history = []
            
            def execute_claude_code(self, prompt, **kwargs):
                if "trigger_error" in prompt:
                    raise RuntimeError("Simulated error")
                return {"status": "success"}
            
            async def validate_code(self, code):
                if "trigger_async_error" in code:
                    raise asyncio.TimeoutError("Validation timeout")
                return {"authentic": True, "confidence": 0.9, "issues": [], "suggestions": []}
        
        return MockAIInterface()
    
    def test_error_propagation(self, ai_interface):
        """Test that errors are properly propagated."""
        # Act & Assert
        with pytest.raises(RuntimeError, match="Simulated error"):
            ai_interface.execute_claude_code("trigger_error test prompt")
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self, ai_interface):
        """Test async error handling."""
        # Act & Assert
        with pytest.raises(asyncio.TimeoutError, match="Validation timeout"):
            await ai_interface.validate_code("trigger_async_error in code")
    
    @patch('subprocess.run')
    def test_subprocess_error_details(self, mock_run):
        """Test detailed subprocess error reporting."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='',
            stderr='Detailed error message with context',
            returncode=2
        )
        
        interface = MockAIInterface()
        
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            interface.execute_claude_code("test prompt")
        
        assert "Detailed error message with context" in str(exc_info.value)
    
    @pytest.mark.parametrize("return_code,expected_error", [
        (1, "Claude Code execution failed"),
        (2, "Claude Code execution failed"),
        (127, "Claude Code execution failed"),
    ])
    @patch('subprocess.run')
    def test_different_error_codes(self, mock_run, return_code, expected_error):
        """Test handling of different subprocess return codes."""
        # Arrange
        mock_run.return_value = MagicMock(
            stdout='',
            stderr=f'Error code {return_code}',
            returncode=return_code
        )
        
        interface = MockAIInterface()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match=expected_error):
            interface.execute_claude_code("test")
    
    def test_empty_prompt_handling(self, ai_interface):
        """Test handling of empty or None prompts."""
        # Test empty string
        result = ai_interface.execute_claude_code("")
        assert result["status"] == "success"
        
        # Test None (should handle gracefully or raise appropriate error)
        try:
            ai_interface.execute_claude_code(None)
        except TypeError:
            # This is acceptable - None prompt should raise TypeError
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_requests(self, ai_interface):
        """Test concurrent code validation requests."""
        # Arrange
        code_samples = [
            "def func1(): return 1",
            "def func2(): return 2", 
            "def func3(): return 3",
            "def func4(): return 4",
            "def func5(): return 5"
        ]
        
        # Act
        tasks = [ai_interface.validate_code(code) for code in code_samples]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 5
        for result in results:
            assert "authentic" in result
            assert "confidence" in result
            assert isinstance(result["issues"], list)
            assert isinstance(result["suggestions"], list)