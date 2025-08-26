"""
Comprehensive unit tests for Validation Service.

Tests cover:
- Code validation and anti-hallucination measures
- Placeholder detection across multiple languages
- Syntax validation and quality assessment
- Security validation and risk detection
- Semantic analysis and coherence checking
- Response validation for different content types
- Progress authenticity verification
- Performance under various validation loads
"""

import pytest
import asyncio
import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from services.validation_service import ValidationService, ValidationLevel, ValidationCategory
from core.exceptions import PlaceholderDetectionError, SemanticValidationError, ValidationError


class TestValidationServiceInitialization:
    """Test Validation Service initialization and setup."""
    
    @pytest.mark.unit
    async def test_service_initialization_success(self, mock_progress_validator):
        """Test successful service initialization."""
        service = ValidationService()
        
        with patch('services.validation_service.ProgressValidator', return_value=mock_progress_validator):
            await service.initialize()
            
            assert service._initialized is True
            assert service._progress_validator is not None
            assert isinstance(service._validation_rules, dict)
            assert isinstance(service._placeholder_patterns, dict)
            assert len(service._validation_rules) > 0
            assert len(service._placeholder_patterns) > 0
    
    @pytest.mark.unit
    async def test_service_initialization_failure(self):
        """Test service initialization failure."""
        service = ValidationService()
        
        with patch('services.validation_service.ProgressValidator', side_effect=Exception("Validator init failed")):
            with pytest.raises(ValidationError) as excinfo:
                await service.initialize()
            
            assert "Validation service initialization failed" in str(excinfo.value)
    
    @pytest.mark.unit
    async def test_health_check(self, validation_service):
        """Test validation service health check."""
        health = await validation_service.health_check()
        
        assert isinstance(health, dict)
        assert health['service'] == 'ValidationService'
        assert health['status'] == 'healthy'
        assert 'progress_validator_available' in health
        assert 'validation_rules_loaded' in health
        assert 'placeholder_patterns_loaded' in health
        assert 'cache_size' in health
        assert 'validation_history_size' in health
    
    @pytest.mark.unit
    async def test_validation_rules_loading(self, validation_service):
        """Test that validation rules are properly loaded."""
        rules = validation_service._validation_rules
        
        assert 'python' in rules
        assert 'javascript' in rules
        assert 'text' in rules
        assert 'general' in rules
        
        # Check Python rules structure
        python_rules = rules['python']
        assert 'forbidden_patterns' in python_rules
        assert 'quality_metrics' in python_rules
        assert 'eval(' in python_rules['forbidden_patterns']
    
    @pytest.mark.unit
    async def test_placeholder_patterns_loading(self, validation_service):
        """Test that placeholder patterns are properly loaded."""
        patterns = validation_service._placeholder_patterns
        
        assert 'python' in patterns
        assert 'javascript' in patterns
        assert 'general' in patterns
        
        # Check pattern format
        python_patterns = patterns['python']
        assert isinstance(python_patterns, list)
        assert len(python_patterns) > 0
        assert any('TODO' in pattern for pattern in python_patterns)


class TestCodeValidation:
    """Test code validation functionality."""
    
    @pytest.mark.unit
    async def test_validate_valid_python_code(self, validation_service, sample_python_code):
        """Test validation of valid Python code."""
        result = await validation_service.validate_code(
            code=sample_python_code,
            language='python',
            validation_level=ValidationLevel.STANDARD
        )
        
        assert result['is_valid'] is True
        assert result['score'] > 0.8
        assert len(result['issues']) == 0
        assert 'categories' in result
        assert 'syntax' in result['categories']
        assert result['categories']['syntax']['is_valid'] is True
    
    @pytest.mark.unit
    async def test_validate_invalid_python_code(self, validation_service, sample_invalid_python_code):
        """Test validation of invalid Python code."""
        result = await validation_service.validate_code(
            code=sample_invalid_python_code,
            language='python',
            validation_level=ValidationLevel.STANDARD
        )
        
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert len(result['issues']) > 0
        assert 'syntax' in result['categories']
        assert result['categories']['syntax']['is_valid'] is False
    
    @pytest.mark.unit
    async def test_validate_code_with_placeholders(self, validation_service):
        """Test validation detects placeholder patterns."""
        code_with_placeholders = '''
def incomplete_function():
    # TODO: Implement this function
    pass  # implement later
    
def another_function():
    ...  # placeholder
    raise NotImplementedError("Fix this")
'''
        
        result = await validation_service.validate_code(
            code=code_with_placeholders,
            language='python',
            check_placeholders=True
        )
        
        assert 'placeholder' in result['categories']
        placeholder_result = result['categories']['placeholder']
        assert placeholder_result['count'] > 0
        assert len(placeholder_result['patterns_found']) > 0
        assert len(placeholder_result['locations']) > 0
        assert placeholder_result['severity'] > 0
    
    @pytest.mark.unit
    async def test_validate_javascript_code(self, validation_service):
        """Test JavaScript code validation."""
        js_code = '''
function calculateSum(a, b) {
    return a + b;
}

const multiply = (x, y) => x * y;

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push({operation: 'add', result});
        return result;
    }
}
'''
        
        result = await validation_service.validate_code(
            code=js_code,
            language='javascript',
            validation_level=ValidationLevel.STANDARD
        )
        
        assert result['is_valid'] is True
        assert result['score'] > 0.8
        assert result['metadata']['language'] == 'javascript'
    
    @pytest.mark.unit
    async def test_validate_empty_code(self, validation_service):
        """Test validation of empty code."""
        result = await validation_service.validate_code(
            code="",
            language='python'
        )
        
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert any('Code content is empty' in issue for issue in result['issues'])
    
    @pytest.mark.unit
    async def test_validate_code_quality_metrics(self, validation_service):
        """Test code quality metrics validation."""
        # Code with quality issues
        poor_quality_code = '''
def very_long_function_with_extremely_long_name_that_exceeds_reasonable_limits():
    x = 1
    # Very long line that exceeds maximum line length limit and should trigger a quality warning in the validation process
    return x

# No docstrings, poor structure
a = 1
b = 2
c = a + b
print(c)
'''
        
        result = await validation_service.validate_code(
            code=poor_quality_code,
            language='python',
            check_quality=True
        )
        
        assert 'quality' in result['categories']
        quality_result = result['categories']['quality']
        assert quality_result['score'] < 1.0
        assert len(quality_result['warnings']) > 0 or len(quality_result['suggestions']) > 0
    
    @pytest.mark.unit
    async def test_validate_code_comprehensive_level(self, validation_service, sample_python_code):
        """Test comprehensive validation level."""
        result = await validation_service.validate_code(
            code=sample_python_code,
            language='python',
            validation_level=ValidationLevel.COMPREHENSIVE
        )
        
        assert 'categories' in result
        assert 'syntax' in result['categories']
        assert 'quality' in result['categories']
        assert 'security' in result['categories']
        assert 'semantic' in result['categories']
    
    @pytest.mark.unit
    async def test_validate_code_security_risks(self, validation_service):
        """Test security risk detection."""
        risky_code = '''
import subprocess

def dangerous_function(user_input):
    # High risk patterns
    eval(user_input)
    exec("print('hello')")
    
    # Medium risk patterns
    password = "hardcoded_password"
    api_key = "sk-123456789"
    
    subprocess.call(user_input, shell=True)
'''
        
        result = await validation_service.validate_code(
            code=risky_code,
            language='python',
            validation_level=ValidationLevel.STRICT
        )
        
        assert 'security' in result['categories']
        security_result = result['categories']['security']
        assert security_result['risk_level'] == 'high'
        assert len(security_result['issues']) > 0
        assert len(security_result['patterns_detected']) > 0


class TestPlaceholderDetection:
    """Test placeholder detection functionality."""
    
    @pytest.mark.unit
    async def test_detect_python_placeholders(self, validation_service):
        """Test Python placeholder pattern detection."""
        python_placeholders = '''
def todo_function():
    # TODO: Implement this
    pass
    
def fixme_function():
    # FIXME: Fix the logic here
    return None  # implement this
    
class PlaceholderClass:
    pass  # Not implemented
    
def ellipsis_function():
    ...  # placeholder
'''
        
        placeholder_result = await validation_service._detect_placeholders(
            python_placeholders, 'python'
        )
        
        assert placeholder_result['count'] > 0
        assert len(placeholder_result['patterns_found']) > 0
        assert len(placeholder_result['locations']) > 0
        assert placeholder_result['severity'] > 0
    
    @pytest.mark.unit
    async def test_detect_javascript_placeholders(self, validation_service):
        """Test JavaScript placeholder pattern detection."""
        js_placeholders = '''
function todoFunction() {
    // TODO: Implement this function
    throw new Error("Not implemented");
}

const fixmeFunction = () => {
    // FIXME: Complete implementation
    console.log("TODO: Add logic here");
};

function placeholderFunction() {
    /* TODO: Add implementation */
}
'''
        
        placeholder_result = await validation_service._detect_placeholders(
            js_placeholders, 'javascript'
        )
        
        assert placeholder_result['count'] > 0
        assert any('TODO' in pattern for pattern in placeholder_result['patterns_found'])
    
    @pytest.mark.unit
    async def test_detect_general_placeholders(self, validation_service):
        """Test general placeholder pattern detection."""
        general_placeholders = '''
This is a document with [PLACEHOLDER] content.
Please FILL_IN_YOUR_NAME_HERE in the form.
The result should be YOUR_VALUE_HERE.
<IMPLEMENT_FEATURE> needs to be completed.
{TODO: Add more details}
'''
        
        placeholder_result = await validation_service._detect_placeholders(
            general_placeholders, 'text'
        )
        
        assert placeholder_result['count'] > 0
        assert len(placeholder_result['patterns_found']) > 0
    
    @pytest.mark.unit
    async def test_placeholder_severity_calculation(self, validation_service):
        """Test placeholder severity calculation."""
        # High severity: many placeholders
        high_severity_code = '''
# TODO: Function 1
def func1(): pass

# TODO: Function 2  
def func2(): pass

# TODO: Function 3
def func3(): pass

# TODO: Function 4
def func4(): pass
'''
        
        result = await validation_service._detect_placeholders(
            high_severity_code, 'python'
        )
        
        assert result['severity'] > 0.5  # Should be high severity
        
        # Low severity: few placeholders
        low_severity_code = '''
def working_function():
    return "This works"

def another_working_function():
    return "This also works"

def mostly_complete():
    # TODO: Add one small feature
    return "Mostly done"
'''
        
        result = await validation_service._detect_placeholders(
            low_severity_code, 'python'
        )
        
        assert result['severity'] < 0.5  # Should be lower severity


class TestSyntaxValidation:
    """Test syntax validation functionality."""
    
    @pytest.mark.unit
    async def test_python_syntax_validation_valid(self, validation_service, sample_python_code):
        """Test Python syntax validation with valid code."""
        syntax_result = await validation_service._validate_syntax(
            sample_python_code, 'python'
        )
        
        assert syntax_result['is_valid'] is True
        assert len(syntax_result['errors']) == 0
    
    @pytest.mark.unit
    async def test_python_syntax_validation_invalid(self, validation_service):
        """Test Python syntax validation with invalid code."""
        invalid_code = '''
def broken_function(
    # Missing closing parenthesis and colon
    return "This won't compile"

def another_broken():
    if True
        print("Missing colon")
'''
        
        syntax_result = await validation_service._validate_syntax(
            invalid_code, 'python'
        )
        
        assert syntax_result['is_valid'] is False
        assert len(syntax_result['errors']) > 0
    
    @pytest.mark.unit
    async def test_javascript_syntax_validation(self, validation_service):
        """Test JavaScript syntax validation (simplified)."""
        # Test with potentially problematic JS
        js_with_syntax_error = '''
function test() {
    if (true {
        console.log("Missing closing parenthesis");
    }
}
'''
        
        syntax_result = await validation_service._validate_syntax(
            js_with_syntax_error, 'javascript'
        )
        
        # Note: This is a simplified check, real implementation might be more sophisticated
        assert 'is_valid' in syntax_result
    
    @pytest.mark.unit
    async def test_syntax_validation_exception_handling(self, validation_service):
        """Test syntax validation exception handling."""
        # Test with problematic input that might cause exceptions
        problematic_code = "import sys\nsys.exit()" # Code that tries to exit
        
        syntax_result = await validation_service._validate_syntax(
            problematic_code, 'python'
        )
        
        # Should handle gracefully
        assert isinstance(syntax_result, dict)
        assert 'is_valid' in syntax_result


class TestSemanticValidation:
    """Test semantic validation functionality."""
    
    @pytest.mark.unit
    async def test_python_semantic_validation(self, validation_service):
        """Test Python semantic validation."""
        code_with_semantic_issues = '''
def function_with_issues():
    unused_variable = "This is never used"
    another_unused = 42
    
    used_variable = "This is used"
    return used_variable

def another_function():
    x = 1
    y = 2
    return x  # y is unused
'''
        
        semantic_result = await validation_service._validate_semantics(
            code_with_semantic_issues, 'python', None
        )
        
        assert semantic_result['score'] <= 1.0
        assert 'coherence_metrics' in semantic_result
        assert semantic_result['coherence_metrics']['unused_variables'] > 0
    
    @pytest.mark.unit
    async def test_semantic_validation_with_file_path(self, validation_service, sample_python_code):
        """Test semantic validation with file path context."""
        semantic_result = await validation_service._validate_semantics(
            sample_python_code, 'python', '/test/path/module.py'
        )
        
        assert isinstance(semantic_result, dict)
        assert 'score' in semantic_result
        assert 'coherence_metrics' in semantic_result
    
    @pytest.mark.unit
    async def test_semantic_validation_exception_handling(self, validation_service):
        """Test semantic validation exception handling."""
        malformed_code = "This is not even close to valid Python code @#$%"
        
        semantic_result = await validation_service._validate_semantics(
            malformed_code, 'python', None
        )
        
        assert len(semantic_result['warnings']) > 0
        assert 'Semantic analysis warning' in semantic_result['warnings'][0]


class TestResponseValidation:
    """Test general response validation functionality."""
    
    @pytest.mark.unit
    async def test_validate_text_response(self, validation_service):
        """Test text response validation."""
        text_response = "This is a comprehensive response that provides detailed information."
        
        result = await validation_service.validate_response(
            response=text_response,
            response_type='text',
            validation_criteria={'min_length': 10}
        )
        
        assert result['is_valid'] is True
        assert result['score'] > 0.0
        assert result['metadata']['response_type'] == 'text'
    
    @pytest.mark.unit
    async def test_validate_text_response_too_short(self, validation_service):
        """Test text response validation with insufficient length."""
        short_response = "Short"
        
        result = await validation_service.validate_response(
            response=short_response,
            response_type='text',
            validation_criteria={'min_length': 50}
        )
        
        assert result['is_valid'] is False
        assert any('too short' in issue.lower() for issue in result['issues'])
    
    @pytest.mark.unit
    async def test_validate_json_response_valid(self, validation_service):
        """Test JSON response validation with valid JSON."""
        json_response = '{"name": "test", "value": 42, "active": true}'
        
        result = await validation_service.validate_response(
            response=json_response,
            response_type='json'
        )
        
        assert result['is_valid'] is True
    
    @pytest.mark.unit
    async def test_validate_json_response_invalid(self, validation_service):
        """Test JSON response validation with invalid JSON."""
        invalid_json = '{"name": "test", "value": 42, "active":}'  # Missing value
        
        result = await validation_service.validate_response(
            response=invalid_json,
            response_type='json'
        )
        
        assert result['is_valid'] is False
        assert any('Invalid JSON' in issue for issue in result['issues'])
    
    @pytest.mark.unit
    async def test_validate_code_response(self, validation_service, sample_python_code):
        """Test code response validation."""
        result = await validation_service.validate_response(
            response=sample_python_code,
            response_type='code',
            validation_criteria={'language': 'python'}
        )
        
        assert 'code_validation' in result
        assert result['is_valid'] is True
        assert result['score'] > 0.0
    
    @pytest.mark.unit
    async def test_validate_empty_response(self, validation_service):
        """Test validation of empty response."""
        result = await validation_service.validate_response(
            response="",
            response_type='text'
        )
        
        assert result['is_valid'] is False
        assert any('empty' in issue.lower() for issue in result['issues'])


class TestProgressAuthenticity:
    """Test progress authenticity checking."""
    
    @pytest.mark.unit
    async def test_check_progress_authenticity_success(self, validation_service, temp_directory):
        """Test successful progress authenticity check."""
        test_file = temp_directory / "test_file.py"
        test_file.write_text("def test(): return True")
        
        result = await validation_service.check_progress_authenticity(
            file_path=test_file,
            project_context={'project_type': 'python'}
        )
        
        assert result['is_authentic'] is True
        assert result['authenticity_score'] > 0.0
        assert 'metadata' in result
        assert result['metadata']['file_path'] == str(test_file)
    
    @pytest.mark.unit
    async def test_check_progress_authenticity_failure(self, validation_service, temp_directory):
        """Test progress authenticity check failure."""
        test_file = temp_directory / "nonexistent.py"
        
        with patch.object(validation_service._progress_validator, 'validate_progress') as mock_validate:
            mock_validate.side_effect = Exception("Validation failed")
            
            result = await validation_service.check_progress_authenticity(
                file_path=test_file
            )
            
            assert result['is_authentic'] is False
            assert result['authenticity_score'] == 0.0
            assert len(result['issues']) > 0
    
    @pytest.mark.unit
    async def test_progress_authenticity_without_validator(self):
        """Test progress authenticity check when validator is not initialized."""
        service = ValidationService()
        # Don't initialize service
        
        with pytest.raises(ValidationError) as excinfo:
            await service.check_progress_authenticity("/test/path")
        
        assert "Progress validator not initialized" in str(excinfo.value)


class TestValidationHistory:
    """Test validation history and reporting."""
    
    @pytest.mark.unit
    async def test_validation_history_tracking(self, validation_service, sample_python_code):
        """Test that validation history is properly tracked."""
        initial_history_size = len(validation_service._validation_history)
        
        await validation_service.validate_code(sample_python_code, 'python')
        
        assert len(validation_service._validation_history) > initial_history_size
        
        latest_entry = validation_service._validation_history[-1]
        assert latest_entry['type'] == 'code'
        assert 'timestamp' in latest_entry
        assert 'is_valid' in latest_entry
        assert 'score' in latest_entry
    
    @pytest.mark.unit
    async def test_validation_report_generation(self, validation_service):
        """Test comprehensive validation report generation."""
        # Perform various validations
        await validation_service.validate_code("def test(): pass", 'python')
        await validation_service.validate_response("Test response", 'text')
        
        # Generate report
        report = await validation_service.get_validation_report()
        
        assert 'total_validations' in report
        assert 'successful_validations' in report
        assert 'success_rate' in report
        assert 'average_score' in report
        assert 'type_breakdown' in report
        assert 'recent_history' in report
        assert report['success_rate'] >= 0.0 and report['success_rate'] <= 1.0
    
    @pytest.mark.unit
    async def test_validation_report_filtering(self, validation_service):
        """Test validation report with filtering."""
        # Perform different types of validation
        await validation_service.validate_code("def test(): pass", 'python')
        await validation_service.validate_response("Test", 'text')
        
        # Filter by type
        code_report = await validation_service.get_validation_report(
            validation_type_filter='code'
        )
        
        assert 'type_breakdown' in code_report
        # Should mainly contain code validations
        
        # Filter by limit
        limited_report = await validation_service.get_validation_report(limit=1)
        assert len(limited_report['recent_history']) <= 1
    
    @pytest.mark.unit
    async def test_validation_history_overflow_handling(self, validation_service):
        """Test validation history overflow management."""
        # Simulate full history
        validation_service._validation_history = [
            {'type': 'test', 'timestamp': f'2025-01-{i:02d}', 'is_valid': True, 'score': 1.0, 'issues_count': 0, 'warnings_count': 0, 'metadata': {}}
            for i in range(1, 51)
        ] * 21  # More than 1000 entries
        
        # Add new validation
        await validation_service.validate_response("Test", 'text')
        
        # History should be trimmed
        assert len(validation_service._validation_history) <= 1000


class TestValidationServiceErrorHandling:
    """Test Validation Service error handling."""
    
    @pytest.mark.unit
    async def test_validation_with_malformed_input(self, validation_service):
        """Test validation with malformed input."""
        # Test with None input
        result = await validation_service.validate_code(
            code=None,
            language='python'
        )
        
        assert result['is_valid'] is False
        
        # Test with non-string input
        result = await validation_service.validate_response(
            response={'malformed': 'object'},
            response_type='text'
        )
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    async def test_validation_exception_handling(self, validation_service):
        """Test handling of validation exceptions."""
        # Mock AST parsing to raise exception
        with patch('ast.parse', side_effect=Exception("Parsing failed")):
            result = await validation_service.validate_code(
                code="def test(): pass",
                language='python'
            )
            
            assert result['is_valid'] is False
            assert len(result['issues']) > 0
    
    @pytest.mark.unit  
    async def test_progress_validator_exception_handling(self, validation_service, temp_directory):
        """Test handling of progress validator exceptions."""
        test_file = temp_directory / "test.py"
        test_file.write_text("def test(): pass")
        
        with patch.object(validation_service._progress_validator, 'validate_progress') as mock_validate:
            mock_validate.side_effect = Exception("Validator crashed")
            
            result = await validation_service.check_progress_authenticity(test_file)
            
            assert result['is_authentic'] is False
            assert 'error' in result['metadata']


class TestValidationServiceEdgeCases:
    """Test Validation Service edge cases and boundary conditions."""
    
    @pytest.mark.edge_case
    async def test_validate_very_large_code(self, validation_service):
        """Test validation of very large code files."""
        # Generate large code sample
        large_code = '''
def large_function():
    """A very large function for testing."""
    # Generate many lines of code
''' + '\n'.join([f'    variable_{i} = {i}' for i in range(5000)]) + '''
    return sum([variable_0, variable_1, variable_2])
'''
        
        result = await validation_service.validate_code(
            large_code,
            'python',
            validation_level=ValidationLevel.BASIC  # Use basic to avoid long processing
        )
        
        assert isinstance(result, dict)
        assert 'metadata' in result
        assert result['metadata']['line_count'] > 5000
    
    @pytest.mark.edge_case
    async def test_validate_unicode_code(self, validation_service):
        """Test validation of code with Unicode characters."""
        unicode_code = '''
def greet_世界():
    """Greet the 世界 (world)."""
    message = "Hello 世界! 🌍"
    print(f"ñoño: {message}")
    return message

class Café:
    """A café class with émojis ☕."""
    def __init__(self):
        self.name = "Café del ñoño"
'''
        
        result = await validation_service.validate_code(
            unicode_code,
            'python'
        )
        
        assert result['is_valid'] is True
        assert '世界' in unicode_code  # Verify Unicode is preserved
    
    @pytest.mark.edge_case
    async def test_validate_code_with_special_characters(self, validation_service):
        """Test validation of code with special characters and symbols."""
        special_code = '''
def special_function():
    """Function with special chars: @#$%^&*()"""
    regex_pattern = r"[a-zA-Z0-9@#$%^&*()_+=-]+"
    special_dict = {"key@#": "value$%", "key^&": "value*()"}
    return regex_pattern, special_dict
'''
        
        result = await validation_service.validate_code(
            special_code,
            'python'
        )
        
        assert result['is_valid'] is True
    
    @pytest.mark.edge_case
    async def test_validate_mixed_language_content(self, validation_service):
        """Test validation of content mixing multiple languages."""
        mixed_content = '''
# Python code
def python_function():
    pass

/* JavaScript code */
function javascriptFunction() {
    return "mixed";
}

// TODO: This is confusing but should be handled
'''
        
        result = await validation_service.validate_code(
            mixed_content,
            'python'  # Validate as Python
        )
        
        # Should detect syntax issues due to JS syntax in Python context
        assert result['is_valid'] is False
    
    @pytest.mark.performance
    async def test_concurrent_validations(self, validation_service, performance_test_config):
        """Test concurrent validation operations."""
        from tests.conftest import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Create validation tasks
        validation_tasks = []
        for i in range(performance_test_config['concurrent_operations']):
            code = f'def function_{i}():\n    return "{i}"'
            task = validation_service.validate_code(code, 'python')
            validation_tasks.append(task)
        
        monitor.start()
        
        results = await asyncio.gather(*validation_tasks)
        
        monitor.stop()
        
        assert len(results) == performance_test_config['concurrent_operations']
        assert all(result['is_valid'] for result in results)
        
        # Performance assertion
        monitor.assert_performance(performance_test_config['max_execution_time'])
    
    @pytest.mark.edge_case
    async def test_validation_with_binary_content(self, validation_service):
        """Test validation handling of binary content."""
        # Simulate binary content (non-UTF8 bytes as string)
        try:
            binary_content = b'\x80\x81\x82\x83'.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            binary_content = "����"  # Replacement characters
        
        result = await validation_service.validate_code(
            binary_content,
            'python'
        )
        
        # Should handle gracefully without crashing
        assert isinstance(result, dict)
        assert 'is_valid' in result
    
    @pytest.mark.edge_case
    async def test_validation_memory_efficiency(self, validation_service):
        """Test validation memory efficiency with repeated operations."""
        # Perform many validations to test memory usage
        initial_history_size = len(validation_service._validation_history)
        
        for i in range(100):
            await validation_service.validate_code(f"def func_{i}(): pass", 'python')
        
        # Ensure history doesn't grow infinitely
        final_history_size = len(validation_service._validation_history)
        growth = final_history_size - initial_history_size
        
        assert growth <= 100  # Should be reasonable growth
    
    @pytest.mark.edge_case
    async def test_deeply_nested_code_structure(self, validation_service):
        """Test validation of deeply nested code structures."""
        nested_code = 'if True:\n'
        for i in range(50):  # Deep nesting
            nested_code += '    ' * (i + 1) + 'if True:\n'
        nested_code += '    ' * 51 + 'pass\n'
        
        result = await validation_service.validate_code(
            nested_code,
            'python'
        )
        
        # Should handle deep nesting without stack overflow
        assert isinstance(result, dict)
        assert result['is_valid'] is True