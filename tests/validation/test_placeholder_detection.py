"""
Unit tests for placeholder detection and anti-hallucination validation.

This module tests the core functionality that prevents AI from generating
placeholder code and ensures authentic implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple


class TestPlaceholderDetection:
    """Test suite for placeholder detection."""
    
    @pytest.fixture
    def detector(self):
        """Create placeholder detector instance."""
        # This will be: from claude_tiu.core.validators import PlaceholderDetector
        
        class MockPlaceholderDetector:
            def __init__(self):
                self.patterns = {
                    'todo_comments': r'#\s*(?:TODO|FIXME|XXX|HACK)',
                    'not_implemented': r'raise\s+NotImplementedError',
                    'pass_statements': r'^\s*pass\s*(?:#.*)?$',
                    'console_log': r'console\.log\s*\(',
                    'placeholder_text': r'(?:placeholder|implement\s+later|fix\s+me)',
                    'empty_functions': r'def\s+\w+\([^)]*\):\s*(?:#[^\n]*)?\s*pass\s*$',
                    'javascript_like': r'(?:alert\s*\(|document\.|window\.)',
                }
            
            def has_placeholder(self, code: str) -> bool:
                """Check if code contains placeholders."""
                for pattern in self.patterns.values():
                    if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                        return True
                return False
            
            def detect_placeholders(self, code: str) -> List[Dict[str, Any]]:
                """Detect all placeholders with details."""
                placeholders = []
                lines = code.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern_name, pattern in self.patterns.items():
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            placeholders.append({
                                'type': pattern_name,
                                'line': i,
                                'column': match.start(),
                                'text': match.group(),
                                'full_line': line.strip(),
                                'severity': self._get_severity(pattern_name)
                            })
                
                return placeholders
            
            def scan_directory(self, directory: Path) -> Dict[str, Dict[str, Any]]:
                """Scan directory for placeholders."""
                results = {}
                
                for py_file in directory.rglob('*.py'):
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        placeholders = self.detect_placeholders(content)
                        
                        results[str(py_file.relative_to(directory))] = {
                            'has_placeholders': len(placeholders) > 0,
                            'placeholder_count': len(placeholders),
                            'placeholders': placeholders,
                            'quality_score': self._calculate_quality_score(content, placeholders)
                        }
                    except Exception as e:
                        results[str(py_file.relative_to(directory))] = {
                            'error': str(e),
                            'has_placeholders': True,  # Assume worst case
                            'placeholder_count': -1
                        }
                
                return results
            
            def _get_severity(self, pattern_name: str) -> str:
                """Get severity level for pattern type."""
                severity_map = {
                    'todo_comments': 'medium',
                    'not_implemented': 'high',
                    'pass_statements': 'high',
                    'console_log': 'high',
                    'placeholder_text': 'medium',
                    'empty_functions': 'high',
                    'javascript_like': 'high'
                }
                return severity_map.get(pattern_name, 'low')
            
            def _calculate_quality_score(self, code: str, placeholders: List[Dict]) -> float:
                """Calculate code quality score (0-1)."""
                if not placeholders:
                    return 1.0
                
                total_lines = len([line for line in code.split('\n') if line.strip()])
                if total_lines == 0:
                    return 0.0
                
                # Weight by severity
                penalty = 0
                for placeholder in placeholders:
                    severity_weights = {'low': 0.1, 'medium': 0.2, 'high': 0.3}
                    penalty += severity_weights.get(placeholder['severity'], 0.2)
                
                # Normalize by total lines
                score = max(0.0, 1.0 - (penalty / total_lines))
                return round(score, 2)
        
        return MockPlaceholderDetector()
    
    def test_todo_comment_detection(self, detector):
        """Test detection of TODO comments."""
        test_cases = [
            ("# TODO: implement this", True),
            ("# todo: fix this bug", True),
            ("# FIXME: handle edge case", True),
            ("# XXX: temporary hack", True),
            ("# HACK: quick fix", True),
            ("# This is a regular comment", False),
            ("def func(): pass  # TODO: complete", True),
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_not_implemented_detection(self, detector):
        """Test detection of NotImplementedError."""
        test_cases = [
            ("raise NotImplementedError", True),
            ("raise NotImplementedError('Not done')", True),
            ("raise NotImplementedError()", True),
            ("def func():\n    raise NotImplementedError", True),
            ("raise ValueError('Invalid')", False),
            ("# raise NotImplementedError", False),  # Commented out
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_pass_statement_detection(self, detector):
        """Test detection of standalone pass statements."""
        test_cases = [
            ("pass", True),
            ("    pass", True),
            ("pass  # TODO: implement", True),
            ("def func():\n    pass", True),
            ("pass  # This is intentional", True),
            ("password = 'secret'", False),  # Contains 'pass' but not statement
            ("bypass = True", False),
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_console_log_detection(self, detector):
        """Test detection of JavaScript-like console.log."""
        test_cases = [
            ("console.log('debug')", True),
            ("console.log(value)", True),
            ("print(console.log)", False),  # Different context
            ("# console.log commented", False),
            ("console_log = True", False),  # Variable name
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_placeholder_text_detection(self, detector):
        """Test detection of placeholder text."""
        test_cases = [
            ("return 'placeholder'", True),
            ("# implement later", True),
            ("# fix me please", True),
            ("# This needs implementation later", True),
            ("implementation = 'complete'", False),
            ("later_function()", False),
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_javascript_like_detection(self, detector):
        """Test detection of JavaScript-like constructs."""
        test_cases = [
            ("alert('message')", True),
            ("document.getElementById('test')", True),
            ("window.location.href", True),
            ("print('alert: something')", False),
            ("doc_string = 'document'", False),
        ]
        
        for code, expected in test_cases:
            result = detector.has_placeholder(code)
            assert result == expected, f"Failed for code: '{code}'"
    
    def test_placeholder_details(self, detector, code_samples):
        """Test detailed placeholder detection."""
        # Use mixed implementation sample
        code = code_samples['mixed_implementation']
        placeholders = detector.detect_placeholders(code)
        
        # Should find multiple placeholders
        assert len(placeholders) > 0
        
        # Check structure of placeholder details
        for placeholder in placeholders:
            assert 'type' in placeholder
            assert 'line' in placeholder
            assert 'column' in placeholder
            assert 'text' in placeholder
            assert 'severity' in placeholder
            assert isinstance(placeholder['line'], int)
            assert placeholder['line'] > 0
    
    def test_quality_score_calculation(self, detector, code_samples):
        """Test quality score calculation."""
        # Test complete function (should have high score)
        complete_code = code_samples['complete_function']
        complete_placeholders = detector.detect_placeholders(complete_code)
        complete_score = detector._calculate_quality_score(complete_code, complete_placeholders)
        assert complete_score >= 0.8
        
        # Test code with placeholders (should have lower score)
        placeholder_code = code_samples['function_with_todo']
        placeholder_placeholders = detector.detect_placeholders(placeholder_code)
        placeholder_score = detector._calculate_quality_score(placeholder_code, placeholder_placeholders)
        assert placeholder_score <= 0.5
    
    def test_scan_directory(self, detector, tmp_path):
        """Test directory scanning functionality."""
        # Create test files
        (tmp_path / "good.py").write_text("""
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
""")
        
        (tmp_path / "bad.py").write_text("""
def subtract(a, b):
    \"\"\"Subtract two numbers.\"\"\"
    # TODO: implement subtraction
    pass
""")
        
        (tmp_path / "mixed.py").write_text("""
def multiply(a, b):
    return a * b

def divide(a, b):
    # FIXME: handle division by zero
    raise NotImplementedError()
""")
        
        # Scan directory
        results = detector.scan_directory(tmp_path)
        
        # Verify results
        assert "good.py" in results
        assert "bad.py" in results
        assert "mixed.py" in results
        
        # Check good file
        assert results["good.py"]["has_placeholders"] is False
        assert results["good.py"]["placeholder_count"] == 0
        assert results["good.py"]["quality_score"] >= 0.8
        
        # Check bad file
        assert results["bad.py"]["has_placeholders"] is True
        assert results["bad.py"]["placeholder_count"] > 0
        
        # Check mixed file
        assert results["mixed.py"]["has_placeholders"] is True
        assert results["mixed.py"]["placeholder_count"] >= 2  # FIXME + NotImplementedError
    
    @pytest.mark.parametrize("pattern_data", [
        pytest.param(pattern, id=pattern.description) 
        for pattern in ValidationFixtures.get_placeholder_patterns()
    ])
    def test_placeholder_patterns_comprehensive(self, detector, pattern_data, validation_fixtures):
        """Test comprehensive placeholder patterns."""
        # This uses the fixture data
        result = detector.has_placeholder(pattern_data.code_sample)
        assert result == pattern_data.expected_match
    
    def test_multiline_placeholder_detection(self, detector):
        """Test detection across multiple lines."""
        multiline_code = '''
def complex_function():
    """A complex function with issues."""
    data = load_data()
    
    # TODO: validate input data
    if not data:
        pass  # TODO: handle empty data
    
    # Process data
    result = []
    for item in data:
        # implement later - complex processing
        console.log(f"Processing {item}")
        result.append(item)
    
    return result
'''
        
        result = detector.has_placeholder(multiline_code)
        assert result is True
        
        placeholders = detector.detect_placeholders(multiline_code)
        assert len(placeholders) >= 4  # Multiple different types
        
        # Verify different types are detected
        detected_types = {p['type'] for p in placeholders}
        assert 'todo_comments' in detected_types
        assert 'pass_statements' in detected_types
        assert 'placeholder_text' in detected_types
        assert 'console_log' in detected_types
    
    def test_false_positive_prevention(self, detector):
        """Test prevention of false positives."""
        legitimate_code = '''
class DocumentProcessor:
    """Process documents with proper implementation."""
    
    def __init__(self):
        self.console = Logger()  # Not console.log
        self.implementation_date = datetime.now()
    
    def process_document(self, doc):
        """Process a document properly."""
        if not doc.is_valid():
            raise ValueError("Invalid document")
        
        # This is a legitimate comment, not TODO
        logger.info(f"Processing document: {doc.name}")
        
        return {
            'status': 'processed',
            'content': doc.content.strip(),
            'metadata': {
                'processed_at': datetime.now(),
                'implementation_version': '1.0'
            }
        }
    
    def validate_password(self, password):
        """Validate password - legitimate use of 'pass' in context."""
        return len(password) >= 8
'''
        
        result = detector.has_placeholder(legitimate_code)
        assert result is False
        
        placeholders = detector.detect_placeholders(legitimate_code)
        assert len(placeholders) == 0
    
    def test_severity_classification(self, detector):
        """Test severity classification of different placeholder types."""
        high_severity_code = '''
def critical_function():
    raise NotImplementedError()
    pass
    console.log("debug")
'''
        
        placeholders = detector.detect_placeholders(high_severity_code)
        
        # Should have multiple high severity placeholders
        high_severity_count = sum(1 for p in placeholders if p['severity'] == 'high')
        assert high_severity_count >= 2
        
        medium_severity_code = '''
def function_with_todos():
    # TODO: implement this
    # implement later
    return None
'''
        
        placeholders = detector.detect_placeholders(medium_severity_code)
        medium_severity_count = sum(1 for p in placeholders if p['severity'] == 'medium')
        assert medium_severity_count >= 1


class TestProgressValidation:
    """Test suite for progress validation."""
    
    @pytest.fixture
    def validator(self):
        """Create progress validator instance."""
        # This will be: from claude_tiu.core.validators import ProgressValidator
        
        class MockProgressValidator:
            def __init__(self, ai_interface=None, code_analyzer=None):
                self.ai_interface = ai_interface or Mock()
                self.code_analyzer = code_analyzer or Mock()
            
            async def calculate_progress(self, project_path: str) -> Dict[str, Any]:
                """Calculate real vs fake progress."""
                # Mock code analysis
                analysis = self.code_analyzer.analyze(project_path)
                
                total_functions = analysis.get('total_functions', 10)
                implemented = analysis.get('implemented_functions', 7)
                placeholders = analysis.get('placeholder_functions', 3)
                
                real_progress = (implemented / total_functions) * 100 if total_functions > 0 else 0
                fake_progress = (placeholders / total_functions) * 100 if total_functions > 0 else 0
                
                return {
                    'real_progress': real_progress,
                    'fake_progress': fake_progress,
                    'total_functions': total_functions,
                    'implemented_functions': implemented,
                    'placeholder_functions': placeholders,
                    'quality_score': max(0.0, real_progress / 100.0),
                    'confidence': 0.9
                }
            
            async def cross_validate_with_ai(self, code: str) -> Dict[str, Any]:
                """Cross-validate with AI."""
                result = await self.ai_interface.validate_code(code)
                
                return {
                    'is_authentic': result.get('authentic', True),
                    'confidence': result.get('confidence', 0.9),
                    'issues': result.get('issues', []),
                    'suggestions': result.get('suggestions', [])
                }
            
            def analyze_function_implementations(self, code: str) -> Tuple[int, int, int]:
                """Analyze function implementations in code."""
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    return 0, 0, 0
                
                total = 0
                implemented = 0
                placeholder = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total += 1
                        
                        # Simple heuristic: check function body
                        has_implementation = False
                        has_placeholder = False
                        
                        for child in ast.walk(node):
                            if isinstance(child, ast.Pass):
                                has_placeholder = True
                            elif isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                                if hasattr(child.exc.func, 'id') and child.exc.func.id == 'NotImplementedError':
                                    has_placeholder = True
                            elif isinstance(child, (ast.Return, ast.Assign, ast.If, ast.For, ast.While)):
                                if child != node:  # Not the function def itself
                                    has_implementation = True
                        
                        if has_placeholder:
                            placeholder += 1
                        elif has_implementation:
                            implemented += 1
                        else:
                            # Empty function
                            placeholder += 1
                
                return total, implemented, placeholder
        
        return MockProgressValidator()
    
    @pytest.mark.asyncio
    async def test_calculate_progress_basic(self, validator):
        """Test basic progress calculation."""
        # Arrange
        validator.code_analyzer.analyze.return_value = {
            'total_functions': 10,
            'implemented_functions': 7,
            'placeholder_functions': 3
        }
        
        # Act
        result = await validator.calculate_progress("/fake/project")
        
        # Assert
        assert result['real_progress'] == 70.0
        assert result['fake_progress'] == 30.0
        assert result['total_functions'] == 10
        assert result['quality_score'] == 0.7
    
    @pytest.mark.asyncio
    async def test_calculate_progress_all_complete(self, validator):
        """Test progress calculation with all functions complete."""
        # Arrange
        validator.code_analyzer.analyze.return_value = {
            'total_functions': 5,
            'implemented_functions': 5,
            'placeholder_functions': 0
        }
        
        # Act
        result = await validator.calculate_progress("/fake/project")
        
        # Assert
        assert result['real_progress'] == 100.0
        assert result['fake_progress'] == 0.0
        assert result['quality_score'] == 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_progress_all_placeholders(self, validator):
        """Test progress calculation with all placeholders."""
        # Arrange
        validator.code_analyzer.analyze.return_value = {
            'total_functions': 8,
            'implemented_functions': 0,
            'placeholder_functions': 8
        }
        
        # Act
        result = await validator.calculate_progress("/fake/project")
        
        # Assert
        assert result['real_progress'] == 0.0
        assert result['fake_progress'] == 100.0
        assert result['quality_score'] == 0.0
    
    @pytest.mark.asyncio
    async def test_cross_validate_authentic_code(self, validator, mock_ai_interface):
        """Test AI cross-validation for authentic code."""
        # Arrange
        validator.ai_interface = mock_ai_interface
        code = "def add(a, b): return a + b"
        
        mock_ai_interface.validate_code = AsyncMock(return_value={
            'authentic': True,
            'confidence': 0.95,
            'issues': [],
            'suggestions': []
        })
        
        # Act
        result = await validator.cross_validate_with_ai(code)
        
        # Assert
        assert result['is_authentic'] is True
        assert result['confidence'] == 0.95
        assert len(result['issues']) == 0
    
    @pytest.mark.asyncio
    async def test_cross_validate_placeholder_code(self, validator, mock_ai_interface):
        """Test AI cross-validation for placeholder code."""
        # Arrange
        validator.ai_interface = mock_ai_interface
        code = "def add(a, b): pass  # TODO: implement"
        
        mock_ai_interface.validate_code = AsyncMock(return_value={
            'authentic': False,
            'confidence': 0.2,
            'issues': ['Contains TODO comment', 'Function not implemented'],
            'suggestions': ['Replace TODO with actual implementation']
        })
        
        # Act
        result = await validator.cross_validate_with_ai(code)
        
        # Assert
        assert result['is_authentic'] is False
        assert result['confidence'] == 0.2
        assert 'Contains TODO comment' in result['issues']
        assert len(result['suggestions']) > 0
    
    def test_analyze_function_implementations(self, validator, code_samples):
        """Test function implementation analysis."""
        # Test complete implementation
        complete_code = code_samples['complete_function']
        total, implemented, placeholder = validator.analyze_function_implementations(complete_code)
        
        assert total > 0
        assert implemented > 0
        assert placeholder == 0
        
        # Test placeholder code
        placeholder_code = code_samples['function_with_todo']
        total, implemented, placeholder = validator.analyze_function_implementations(placeholder_code)
        
        assert total > 0
        assert implemented == 0
        assert placeholder > 0
    
    def test_analyze_mixed_implementations(self, validator, code_samples):
        """Test analysis of mixed implementation code."""
        mixed_code = code_samples['mixed_implementation']
        total, implemented, placeholder = validator.analyze_function_implementations(mixed_code)
        
        # Should have both implemented and placeholder functions
        assert total > 2
        assert implemented > 0
        assert placeholder > 0
        assert implemented + placeholder <= total  # Some might be empty
    
    @pytest.mark.parametrize("total,impl,placeholder,expected_real,expected_fake", [
        (10, 8, 2, 80.0, 20.0),
        (5, 5, 0, 100.0, 0.0),
        (3, 0, 3, 0.0, 100.0),
        (0, 0, 0, 0.0, 0.0),  # Edge case: no functions
        (1, 1, 0, 100.0, 0.0),  # Single complete function
    ])
    @pytest.mark.asyncio
    async def test_progress_calculation_scenarios(self, validator, total, impl, placeholder, expected_real, expected_fake):
        """Test various progress calculation scenarios."""
        # Arrange
        validator.code_analyzer.analyze.return_value = {
            'total_functions': total,
            'implemented_functions': impl,
            'placeholder_functions': placeholder
        }
        
        # Act
        result = await validator.calculate_progress("/test")
        
        # Assert
        assert abs(result['real_progress'] - expected_real) < 0.1
        assert abs(result['fake_progress'] - expected_fake) < 0.1
    
    @pytest.mark.asyncio
    async def test_validator_error_handling(self, validator):
        """Test error handling in validator."""
        # Arrange - make code analyzer fail
        validator.code_analyzer.analyze.side_effect = Exception("Analysis failed")
        
        # Act & Assert
        with pytest.raises(Exception, match="Analysis failed"):
            await validator.calculate_progress("/test")
    
    def test_malformed_code_handling(self, validator):
        """Test handling of malformed code."""
        malformed_code = "def incomplete_function(\npass  # syntax error"
        
        # Should handle syntax errors gracefully
        total, implemented, placeholder = validator.analyze_function_implementations(malformed_code)
        
        # Should return zeros for malformed code
        assert total == 0
        assert implemented == 0
        assert placeholder == 0