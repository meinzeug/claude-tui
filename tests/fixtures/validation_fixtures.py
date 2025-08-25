"""
Validation-related test fixtures for anti-hallucination testing.
"""

import pytest
from faker import Faker
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

fake = Faker()


class PlaceholderType(Enum):
    """Types of placeholders that can be detected."""
    TODO_COMMENT = "todo_comment"
    NOT_IMPLEMENTED = "not_implemented"
    PASS_STATEMENT = "pass_statement"
    CONSOLE_LOG = "console_log"
    EMPTY_FUNCTION = "empty_function"
    PLACEHOLDER_TEXT = "placeholder_text"
    IMPLEMENT_LATER = "implement_later"


@dataclass
class PlaceholderPattern:
    """Represents a placeholder pattern for testing."""
    pattern: str
    code_sample: str
    expected_match: bool
    placeholder_type: PlaceholderType
    description: str


@dataclass
class ValidationResult:
    """Validation result structure for testing."""
    real_progress: float
    fake_progress: float
    placeholders: List[str]
    quality_score: float
    auto_fix_available: bool
    total_functions: int
    implemented_functions: int
    placeholder_functions: int
    issues: List[str]
    suggestions: List[str]


class ValidationFixtures:
    """Factory for creating validation test data."""
    
    @staticmethod
    def get_placeholder_patterns() -> List[PlaceholderPattern]:
        """Get comprehensive list of placeholder patterns."""
        return [
            # TODO patterns
            PlaceholderPattern(
                pattern=r"#\s*TODO|#\s*FIXME|#\s*XXX",
                code_sample="def func():\n    # TODO: implement this\n    pass",
                expected_match=True,
                placeholder_type=PlaceholderType.TODO_COMMENT,
                description="TODO/FIXME/XXX comments"
            ),
            
            # NotImplementedError patterns
            PlaceholderPattern(
                pattern=r"raise\s+NotImplementedError",
                code_sample="def func():\n    raise NotImplementedError('Not yet implemented')",
                expected_match=True,
                placeholder_type=PlaceholderType.NOT_IMPLEMENTED,
                description="NotImplementedError exceptions"
            ),
            
            # Pass statements
            PlaceholderPattern(
                pattern=r"^\s*pass\s*$",
                code_sample="def func():\n    pass",
                expected_match=True,
                placeholder_type=PlaceholderType.PASS_STATEMENT,
                description="Standalone pass statements"
            ),
            
            # Console.log patterns (JavaScript-like)
            PlaceholderPattern(
                pattern=r"console\.log\(",
                code_sample="def func():\n    console.log('debug')\n    return None",
                expected_match=True,
                placeholder_type=PlaceholderType.CONSOLE_LOG,
                description="JavaScript console.log in Python"
            ),
            
            # Placeholder text patterns
            PlaceholderPattern(
                pattern=r"placeholder|implement\s+later|fix\s+me",
                code_sample="def func():\n    # implement later\n    return 'placeholder'",
                expected_match=True,
                placeholder_type=PlaceholderType.PLACEHOLDER_TEXT,
                description="Placeholder text and 'implement later' comments"
            ),
            
            # Valid complete code (should not match)
            PlaceholderPattern(
                pattern=r"#\s*TODO",
                code_sample="def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b",
                expected_match=False,
                placeholder_type=PlaceholderType.TODO_COMMENT,
                description="Complete implementation without placeholders"
            )
        ]
    
    @staticmethod
    def get_code_samples() -> Dict[str, str]:
        """Get various code samples for testing."""
        return {
            "complete_function": '''
def calculate_interest(principal, rate, time):
    """Calculate compound interest."""
    if principal <= 0 or rate < 0 or time < 0:
        raise ValueError("Invalid input parameters")
    
    return principal * (1 + rate) ** time
''',
            
            "function_with_todo": '''
def process_payment(amount, method):
    """Process payment with given method."""
    # TODO: implement payment validation
    # TODO: add fraud detection
    pass
''',
            
            "mixed_implementation": '''
def user_service():
    """User service with mixed implementation."""
    
    def get_user(user_id):
        """Complete implementation."""
        return {"id": user_id, "name": "John"}
    
    def create_user(data):
        """Incomplete implementation."""
        # TODO: validate data
        # TODO: save to database
        raise NotImplementedError("User creation not implemented")
    
    def delete_user(user_id):
        """Another incomplete."""
        # implement later
        pass
    
    return {
        "get": get_user,
        "create": create_user,
        "delete": delete_user
    }
''',
            
            "javascript_like_placeholders": '''
def debug_function():
    """Function with JavaScript-like debugging."""
    console.log("Debug info")
    alert("This shouldn't be here")
    document.getElementById("test")
    return "result"
''',
            
            "empty_class": '''
class DataProcessor:
    """Data processor class."""
    
    def __init__(self):
        pass
    
    def process(self, data):
        # TODO: implement processing logic
        pass
    
    def validate(self, data):
        raise NotImplementedError()
''',
            
            "complete_class": '''
class Calculator:
    """Complete calculator implementation."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
''',
            
            "subtle_placeholders": '''
def complex_algorithm():
    """Complex algorithm with subtle placeholders."""
    data = load_data()
    
    # Real implementation
    processed = []
    for item in data:
        if item.is_valid():
            processed.append(item.transform())
    
    # Subtle placeholder
    # optimize this later
    result = processed
    
    return result

def load_data():
    # implement proper data loading
    return []
''',
            
            "async_function": '''
async def fetch_user_data(user_id):
    """Async function with placeholder."""
    async with aiohttp.ClientSession() as session:
        # TODO: implement proper error handling
        # TODO: add retry logic
        async with session.get(f"/api/users/{user_id}") as response:
            return await response.json()
'''
        }
    
    @staticmethod
    def create_validation_result(
        real_progress: float = 70.0,
        fake_progress: float = 30.0,
        **overrides
    ) -> ValidationResult:
        """Create validation result for testing."""
        defaults = {
            "placeholders": [
                "TODO in line 5: implement validation logic",
                "NotImplementedError in line 12: create_user function",
                "pass statement in line 18: empty implementation"
            ],
            "quality_score": max(0.0, min(1.0, real_progress / 100.0)),
            "auto_fix_available": fake_progress > 0,
            "total_functions": 10,
            "implemented_functions": int(real_progress / 10),
            "placeholder_functions": int(fake_progress / 10),
            "issues": [
                "Function has empty implementation",
                "Missing error handling",
                "Incomplete validation logic"
            ],
            "suggestions": [
                "Replace TODO comments with actual implementation",
                "Add proper error handling",
                "Implement missing validation logic",
                "Remove placeholder functions"
            ]
        }
        defaults.update(overrides)
        
        return ValidationResult(
            real_progress=real_progress,
            fake_progress=fake_progress,
            **defaults
        )
    
    @staticmethod
    def create_progress_test_cases() -> List[Tuple[str, float, float]]:
        """Create test cases for progress calculation."""
        return [
            # (code_sample_key, expected_real_progress, expected_fake_progress)
            ("complete_function", 100.0, 0.0),
            ("function_with_todo", 0.0, 100.0),
            ("mixed_implementation", 33.33, 66.67),
            ("complete_class", 100.0, 0.0),
            ("empty_class", 0.0, 100.0),
            ("subtle_placeholders", 60.0, 40.0),
            ("javascript_like_placeholders", 20.0, 80.0),
        ]
    
    @staticmethod
    def create_cross_validation_scenarios() -> List[Dict[str, Any]]:
        """Create scenarios for AI cross-validation testing."""
        return [
            {
                "name": "high_quality_code",
                "code": ValidationFixtures.get_code_samples()["complete_function"],
                "expected_authentic": True,
                "expected_confidence": 0.95,
                "expected_issues": []
            },
            {
                "name": "placeholder_heavy_code",
                "code": ValidationFixtures.get_code_samples()["function_with_todo"],
                "expected_authentic": False,
                "expected_confidence": 0.1,
                "expected_issues": ["Multiple TODO comments", "Empty function body"]
            },
            {
                "name": "mixed_quality_code",
                "code": ValidationFixtures.get_code_samples()["mixed_implementation"],
                "expected_authentic": False,
                "expected_confidence": 0.6,
                "expected_issues": ["Incomplete implementations", "Mixed quality"]
            },
            {
                "name": "subtle_issues",
                "code": ValidationFixtures.get_code_samples()["subtle_placeholders"],
                "expected_authentic": False,
                "expected_confidence": 0.7,
                "expected_issues": ["Subtle placeholder comments"]
            }
        ]


@pytest.fixture
def validation_fixtures():
    """Provide validation fixtures factory."""
    return ValidationFixtures


@pytest.fixture
def placeholder_patterns():
    """Get placeholder patterns for testing."""
    return ValidationFixtures.get_placeholder_patterns()


@pytest.fixture
def code_samples():
    """Get code samples for testing."""
    return ValidationFixtures.get_code_samples()


@pytest.fixture
def validation_result():
    """Create sample validation result."""
    return ValidationFixtures.create_validation_result()


@pytest.fixture
def high_quality_validation_result():
    """Create validation result for high-quality code."""
    return ValidationFixtures.create_validation_result(
        real_progress=95.0,
        fake_progress=5.0,
        placeholders=["Minor TODO in line 42"],
        quality_score=0.95,
        auto_fix_available=True
    )


@pytest.fixture
def poor_quality_validation_result():
    """Create validation result for poor-quality code."""
    return ValidationFixtures.create_validation_result(
        real_progress=20.0,
        fake_progress=80.0,
        placeholders=[
            "TODO in line 5: implement core logic",
            "TODO in line 12: add error handling",
            "NotImplementedError in line 18",
            "pass statement in line 25",
            "console.log in line 30"
        ],
        quality_score=0.2,
        auto_fix_available=True,
        issues=[
            "Multiple incomplete functions",
            "No error handling",
            "JavaScript-like debugging code",
            "Empty implementations"
        ]
    )


@pytest.fixture
def progress_test_cases():
    """Get progress calculation test cases."""
    return ValidationFixtures.create_progress_test_cases()


@pytest.fixture
def cross_validation_scenarios():
    """Get cross-validation test scenarios."""
    return ValidationFixtures.create_cross_validation_scenarios()


class ValidationTestHelper:
    """Helper methods for validation testing."""
    
    @staticmethod
    def assert_valid_progress_result(result: ValidationResult):
        """Assert progress result is valid."""
        assert 0.0 <= result.real_progress <= 100.0, "Real progress must be 0-100"
        assert 0.0 <= result.fake_progress <= 100.0, "Fake progress must be 0-100"
        assert abs(result.real_progress + result.fake_progress - 100.0) < 1.0, "Progress should sum to ~100"
        assert 0.0 <= result.quality_score <= 1.0, "Quality score must be 0-1"
        assert isinstance(result.placeholders, list), "Placeholders must be a list"
        assert isinstance(result.issues, list), "Issues must be a list"
        assert isinstance(result.suggestions, list), "Suggestions must be a list"
    
    @staticmethod
    def assert_placeholder_detection(pattern: PlaceholderPattern, detector_func):
        """Assert placeholder pattern detection works correctly."""
        result = detector_func(pattern.code_sample)
        if pattern.expected_match:
            assert result, f"Should detect placeholder in: {pattern.description}"
        else:
            assert not result, f"Should not detect placeholder in: {pattern.description}"
    
    @staticmethod
    def count_function_implementations(code: str) -> Tuple[int, int, int]:
        """Count total, implemented, and placeholder functions in code."""
        import ast
        import re
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0, 0, 0
        
        total_functions = 0
        implemented_functions = 0
        placeholder_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                
                # Get function source
                lines = code.split('\\n')
                func_start = node.lineno - 1
                func_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                func_source = '\\n'.join(lines[func_start:func_end])
                
                # Check for placeholders
                placeholder_patterns = [
                    r'#\s*TODO',
                    r'#\s*FIXME',
                    r'raise\s+NotImplementedError',
                    r'^\s*pass\s*$',
                    r'console\.log',
                    r'implement\s+later'
                ]
                
                has_placeholder = any(
                    re.search(pattern, func_source, re.IGNORECASE | re.MULTILINE)
                    for pattern in placeholder_patterns
                )
                
                if has_placeholder:
                    placeholder_functions += 1
                else:
                    implemented_functions += 1
        
        return total_functions, implemented_functions, placeholder_functions
    
    @staticmethod
    def create_mock_ai_validation(authentic: bool = True, confidence: float = 0.9) -> Dict[str, Any]:
        """Create mock AI validation response."""
        if authentic:
            return {
                "authentic": True,
                "confidence": confidence,
                "issues": [],
                "suggestions": [],
                "quality_metrics": {
                    "completeness": 0.95,
                    "consistency": 0.92,
                    "maintainability": 0.88
                }
            }
        else:
            return {
                "authentic": False,
                "confidence": confidence,
                "issues": [
                    "Contains placeholder implementations",
                    "Missing proper error handling",
                    "Incomplete function bodies"
                ],
                "suggestions": [
                    "Replace TODO comments with implementations",
                    "Add comprehensive error handling",
                    "Complete all function implementations"
                ],
                "quality_metrics": {
                    "completeness": 0.3,
                    "consistency": 0.4,
                    "maintainability": 0.2
                }
            }


@pytest.fixture
def validation_helper():
    """Provide validation test helper."""
    return ValidationTestHelper