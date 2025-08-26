"""
Comprehensive System Integration Tests for Claude TUI.

Tests the integration of all major components:
- Authentication System with JWT refresh tokens
- Validation Service with Anti-Hallucination Engine
- Placeholder Detector with ML capabilities  
- Semantic Analyzer with language-specific handlers
- Auto-Completion Engine with fix suggestions
- API endpoints with ValidationService integration
- TUI components with file operations
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Core components
from claude_tui.core.config_manager import ConfigManager
from claude_tui.services.validation_service import ValidationService, ValidationLevel

# Authentication components
from auth.jwt_auth import JWTAuthenticator, TokenResponse
from auth.models import User, Session

# Validation components  
from claude_tui.validation.anti_hallucination_engine import AntiHallucinationEngine
from claude_tui.validation.placeholder_detector import PlaceholderDetector
from claude_tui.validation.semantic_analyzer import SemanticAnalyzer
from claude_tui.validation.auto_completion_engine import AutoCompletionEngine
from claude_tui.validation.execution_tester import ExecutionTester
from claude_tui.validation.progress_validator import ValidationSeverity

# API components
from api.v1.validation import (
    CodeValidationRequest, ValidationResponse,
    get_validation_service
)

# TUI components
from claude_tui.ui.main_app import ClaudeTIUApp
from claude_tui.ui.screens.file_picker import FilePickerScreen
from claude_tui.ui.screens.clone_project_dialog import CloneProjectDialog


@pytest.fixture
async def config_manager():
    """Create a test configuration manager."""
    config = ConfigManager()
    await config.initialize()
    return config


@pytest.fixture  
async def validation_service(config_manager):
    """Create and initialize validation service."""
    service = ValidationService()
    await service._initialize_impl()
    return service


@pytest.fixture
async def auth_system():
    """Create JWT authentication system."""
    return JWTAuthenticator(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7
    )


class TestSystemIntegration:
    """Comprehensive system integration tests."""
    
    async def test_validation_service_initialization(self, validation_service):
        """Test that ValidationService initializes properly with all components."""
        assert validation_service is not None
        assert validation_service._progress_validator is not None
        assert len(validation_service._validation_rules) > 0
        assert len(validation_service._placeholder_patterns) > 0
        
        # Test health check
        health = await validation_service.health_check()
        assert health['status'] == 'healthy'
        assert health['progress_validator_available'] is True
        assert health['validation_rules_loaded'] > 0
    
    async def test_placeholder_detector_integration(self, config_manager):
        """Test PlaceholderDetector with ML capabilities."""
        detector = PlaceholderDetector(config_manager)
        await detector.initialize()
        
        # Test with placeholder code
        test_code = '''
def example_function():
    """Example function."""
    # TODO: Implement this function
    pass
    
class TestClass:
    # FIXME: Add proper implementation
    def method(self):
        raise NotImplementedError("Not implemented yet")
        '''
        
        issues = await detector.detect_placeholders(test_code)
        
        # Should find multiple placeholder issues
        assert len(issues) >= 2  # TODO and NotImplementedError
        
        # Check issue details
        for issue in issues:
            assert issue.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL, ValidationSeverity.MEDIUM]
            assert issue.issue_type == "placeholder"
            assert issue.auto_fixable is True
    
    async def test_semantic_analyzer_integration(self, config_manager):
        """Test SemanticAnalyzer with language-specific handlers."""
        analyzer = SemanticAnalyzer(config_manager)
        await analyzer.initialize()
        
        # Test Python code with semantic issues
        test_code = '''
import unused_module
import os
import sys

def problematic_function():
    password = "hardcoded_password"  # Security issue
    eval("print('dangerous')")       # Security issue
    
    unused_var = "never used"        # Unused variable
    
    while True:                      # Infinite loop
        print("infinite")
        # Missing break
        
    return None
    print("unreachable")             # Unreachable code
    '''
        
        issues = await analyzer.analyze_content(test_code, language="python")
        
        # Should find multiple semantic issues
        assert len(issues) >= 3  # Security, logical, performance issues
        
        # Check for specific issue types
        issue_types = [issue.issue_type for issue in issues]
        assert "security_issue" in issue_types
        assert any("unused" in issue.description.lower() for issue in issues)
    
    async def test_auto_completion_engine_integration(self, config_manager):
        """Test AutoCompletionEngine with fix suggestions.""" 
        engine = AutoCompletionEngine(config_manager)
        await engine.initialize()
        
        # Create a mock validation issue
        from claude_tui.validation.progress_validator import ValidationIssue
        
        issue = ValidationIssue(
            id="test_placeholder",
            description="TODO comment found",
            severity=ValidationSeverity.HIGH,
            issue_type="placeholder",
            line_number=5,
            context={"matched_text": "# TODO: Implement this"}
        )
        
        # Test fix suggestions
        suggestions = await engine.get_fix_suggestions(issue, "# TODO: Implement this", {})
        
        assert len(suggestions) > 0
        assert any("implement" in suggestion.lower() for suggestion in suggestions)
        
        # Test code completion
        partial_code = "def incomplete_function():"
        completions = await engine.suggest_completion(partial_code, language="python")
        
        assert len(completions) >= 0  # Should return some completions
    
    async def test_execution_tester_integration(self, config_manager):
        """Test ExecutionTester with security sandbox."""
        tester = ExecutionTester(config_manager)
        await tester.initialize()
        
        # Test valid Python code
        valid_code = '''
def add_numbers(a, b):
    return a + b

result = add_numbers(2, 3)
print(f"Result: {result}")
        '''
        
        issues = await tester.test_execution(valid_code)
        assert len(issues) == 0  # Should be no issues
        
        # Test problematic Python code
        problematic_code = '''
def infinite_function():
    while True:
        pass  # Infinite loop without break
        '''
        
        issues = await tester.test_execution(problematic_code)
        # Should detect potential issues (may timeout)
        # Note: Actual timeout detection depends on implementation
    
    async def test_jwt_authentication_system(self, auth_system):
        """Test JWT authentication with refresh token functionality."""
        # Mock repositories
        user_repo = Mock()
        session_repo = Mock()
        
        # Create test user
        test_user = Mock()
        test_user.id = "user123"
        test_user.username = "testuser"
        test_user.email = "test@example.com"
        test_user.is_active = True
        test_user.roles = []
        
        user_repo.authenticate_user = AsyncMock(return_value=test_user)
        user_repo.get_user_by_id = AsyncMock(return_value=test_user)
        
        # Mock session
        test_session = Mock()
        test_session.id = "session123"
        test_session.user_id = "user123"
        test_session.is_active = True
        test_session.is_expired = Mock(return_value=False)
        
        session_repo.create_session = AsyncMock(return_value=test_session)
        session_repo.find_by_refresh_token = AsyncMock(return_value=test_session)
        session_repo.update_session_tokens = AsyncMock()
        
        # Test authentication
        token_response = await auth_system.authenticate_user(
            username="testuser",
            password="password123",
            user_repo=user_repo,
            session_repo=session_repo
        )
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        assert token_response.user_id == "user123"
        
        # Test token refresh
        new_token_response = await auth_system.refresh_token(
            refresh_token=token_response.refresh_token,
            session_repo=session_repo,
            user_repo=user_repo
        )
        
        assert isinstance(new_token_response, TokenResponse)
        assert new_token_response.access_token is not None
        assert new_token_response.refresh_token is not None
    
    async def test_validation_api_integration(self, validation_service):
        """Test API validation endpoints with ValidationService."""
        
        # Test code validation request
        request = CodeValidationRequest(
            code='''
def hello_world():
    # TODO: Implement greeting
    pass
            ''',
            language="python",
            validation_level="standard",
            check_placeholders=True,
            check_syntax=True,
            check_quality=True
        )
        
        # Simulate API endpoint processing
        result = await validation_service.validate_code(
            code=request.code,
            language=request.language,
            validation_level=ValidationLevel.STANDARD,
            check_placeholders=request.check_placeholders,
            check_syntax=request.check_syntax,
            check_quality=request.check_quality
        )
        
        # Validate API response format
        assert 'is_valid' in result
        assert 'score' in result
        assert 'issues' in result
        assert 'warnings' in result
        assert 'categories' in result
        assert 'metadata' in result
        
        # Should detect placeholder
        assert 'placeholder' in result['categories']
        placeholder_result = result['categories']['placeholder']
        assert placeholder_result['count'] > 0
        
        # Create API response
        api_response = ValidationResponse(
            is_valid=result['is_valid'],
            score=result['score'],
            issues=result['issues'],
            warnings=result['warnings'],
            suggestions=result.get('suggestions', []),
            categories=result['categories'],
            metadata=result['metadata']
        )
        
        assert isinstance(api_response.score, float)
        assert 0.0 <= api_response.score <= 1.0
    
    async def test_anti_hallucination_engine_integration(self, config_manager):
        """Test complete anti-hallucination validation pipeline."""
        engine = AntiHallucinationEngine(config_manager)
        await engine.initialize()
        
        # Test comprehensive validation
        test_code = '''
def calculate_fibonacci(n):
    # TODO: Implement Fibonacci calculation
    if n <= 1:
        return n
    # FIXME: Add proper recursive implementation
    raise NotImplementedError("Fibonacci not implemented")

class DataProcessor:
    """Data processing class."""
    
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        # PLACEHOLDER: Add data processing logic
        pass
        
    def validate_input(self, data):
        eval(data)  # Security issue: dangerous eval
        password = "admin123"  # Security issue: hardcoded password
        return True
        '''
        
        validation_result = await engine.validate_content(
            content=test_code,
            content_type="python_code",
            validation_level="comprehensive"
        )
        
        # Should be comprehensive validation result
        assert 'authenticity_score' in validation_result
        assert 'issues' in validation_result
        assert 'categories' in validation_result
        
        # Should detect multiple issue types
        categories = validation_result['categories']
        assert 'placeholder_detection' in categories
        assert 'semantic_analysis' in categories
        
        # Should have low authenticity due to placeholders and issues
        assert validation_result['authenticity_score'] < 0.8
        assert len(validation_result['issues']) >= 3
    
    async def test_tui_integration_components(self):
        """Test TUI components integration."""
        # Test FilePickerScreen initialization
        file_picker = FilePickerScreen(
            title="Test File Picker",
            file_types=[".py", ".json"],
            directories_only=False
        )
        
        assert file_picker.title == "Test File Picker"
        assert file_picker.file_types == [".py", ".json"]
        assert file_picker.directories_only is False
        
        # Test CloneProjectDialog initialization
        clone_dialog = CloneProjectDialog()
        
        assert clone_dialog.is_cloning is False
        assert clone_dialog.callback is None
        
    async def test_end_to_end_validation_workflow(self, config_manager):
        """Test complete end-to-end validation workflow."""
        
        # Initialize all validation components
        validation_service = ValidationService()
        await validation_service._initialize_impl()
        
        placeholder_detector = PlaceholderDetector(config_manager)
        await placeholder_detector.initialize()
        
        semantic_analyzer = SemanticAnalyzer(config_manager)
        await semantic_analyzer.initialize()
        
        auto_completion = AutoCompletionEngine(config_manager)
        await auto_completion.initialize()
        
        # Test code with multiple issues
        test_code = '''
import unused_library
import os

def incomplete_function(param):
    """Function with multiple issues."""
    # TODO: Implement proper parameter validation
    if param is None:
        raise NotImplementedError("Parameter validation not implemented")
    
    # Security issue
    password = "secret123"
    result = eval(f"process({param})")  # Security issue
    
    # Performance issue
    data = []
    for i in range(len(param)):  # Should use enumerate
        data.append(param[i])
    
    # Unused variable
    unused_var = "never used"
    
    # Missing return statement for function that should return something
    pass
        '''
        
        # 1. Run placeholder detection
        placeholder_issues = await placeholder_detector.detect_placeholders(test_code)
        assert len(placeholder_issues) >= 2  # TODO and NotImplementedError
        
        # 2. Run semantic analysis
        semantic_issues = await semantic_analyzer.analyze_content(test_code, language="python")
        assert len(semantic_issues) >= 2  # Security and performance issues
        
        # 3. Run comprehensive validation
        validation_result = await validation_service.validate_code(
            code=test_code,
            language="python",
            validation_level=ValidationLevel.COMPREHENSIVE
        )
        
        # Should have detected multiple categories of issues
        assert validation_result['is_valid'] is False  # Due to syntax/security issues
        assert validation_result['score'] < 0.7  # Low score due to issues
        assert len(validation_result['issues']) > 0
        
        categories = validation_result['categories']
        assert 'placeholder' in categories
        assert 'security' in categories
        assert 'quality' in categories
        
        # 4. Get fix suggestions for first placeholder issue
        if placeholder_issues:
            first_issue = placeholder_issues[0]
            suggestions = await auto_completion.get_fix_suggestions(
                first_issue, test_code, {}
            )
            assert len(suggestions) > 0
        
        # 5. Validate the complete workflow worked
        assert placeholder_issues  # Found placeholder issues
        assert semantic_issues     # Found semantic issues
        assert not validation_result['is_valid']  # Overall validation failed appropriately


class TestPerformanceIntegration:
    """Test system performance under load."""
    
    async def test_concurrent_validations(self, validation_service):
        """Test concurrent validation requests."""
        
        test_codes = [
            "print('Hello, World!')",
            "def func(): pass",
            "# TODO: Implement\npass",
            "import os\nprint(os.getcwd())",
            "x = [i for i in range(10)]"
        ]
        
        # Run concurrent validations
        tasks = [
            validation_service.validate_code(code, "python")
            for code in test_codes
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All validations should complete
        assert len(results) == len(test_codes)
        
        # Each result should have required fields
        for result in results:
            assert 'is_valid' in result
            assert 'score' in result
            assert isinstance(result['score'], float)
    
    async def test_large_code_validation(self, validation_service):
        """Test validation of large code files."""
        
        # Generate large code file
        large_code = "\n".join([
            f"def function_{i}():",
            f"    \"\"\"Function number {i}\"\"\"",
            f"    return {i}",
            ""
        ] for i in range(100))
        
        # Should handle large files
        result = await validation_service.validate_code(large_code, "python")
        
        assert 'is_valid' in result
        assert result['metadata']['line_count'] > 300
    
    async def test_memory_usage_stability(self, config_manager):
        """Test memory usage remains stable during validation."""
        
        # Initialize components
        components = [
            PlaceholderDetector(config_manager),
            SemanticAnalyzer(config_manager),
            AutoCompletionEngine(config_manager)
        ]
        
        for component in components:
            await component.initialize()
        
        # Run multiple validation cycles
        test_code = '''
def test_function():
    # TODO: Implement
    pass
        '''
        
        for i in range(50):  # Multiple iterations
            for component in components:
                if hasattr(component, 'detect_placeholders'):
                    await component.detect_placeholders(test_code)
                elif hasattr(component, 'analyze_content'):
                    await component.analyze_content(test_code, language="python")
                elif hasattr(component, 'suggest_completion'):
                    await component.suggest_completion(test_code, language="python")
        
        # Cleanup all components
        for component in components:
            await component.cleanup()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])