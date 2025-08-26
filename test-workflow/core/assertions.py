"""
Assertion Framework - Comprehensive assertion library for test-workflow
Integrates with test runner and provides detailed failure information
"""

import inspect
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
import re
import json
import difflib


class ComparisonType(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    IS = "is"
    IS_NOT = "is_not"
    ISINSTANCE = "isinstance"
    REGEX_MATCH = "regex_match"
    CONTAINS = "contains"


@dataclass
class AssertionResult:
    """Detailed assertion result for integration with test runner"""
    test_name: str
    assertion_type: ComparisonType
    passed: bool
    expected: Any
    actual: Any
    message: str
    traceback_info: str
    line_number: int
    file_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'assertion_type': self.assertion_type.value,
            'passed': self.passed,
            'expected': str(self.expected),
            'actual': str(self.actual),
            'message': self.message,
            'traceback_info': self.traceback_info,
            'line_number': self.line_number,
            'file_name': self.file_name
        }


class AssertionError(Exception):
    """Custom assertion error with detailed information"""
    
    def __init__(
        self, 
        message: str, 
        assertion_result: AssertionResult
    ):
        super().__init__(message)
        self.assertion_result = assertion_result


class AssertionFramework:
    """
    Comprehensive assertion framework that integrates with test runner
    Provides detailed failure information and tracks all assertions
    """
    
    def __init__(self):
        self.results: Dict[str, List[AssertionResult]] = {}
        self.current_test: Optional[str] = None
        
    def reset_for_test(self, test_name: str) -> None:
        """Reset assertion state for new test"""
        self.current_test = test_name
        self.results[test_name] = []
        
    def get_results(self, test_name: str) -> List[AssertionResult]:
        """Get all assertion results for a test"""
        return self.results.get(test_name, [])
        
    def _record_assertion(
        self,
        assertion_type: ComparisonType,
        passed: bool,
        expected: Any,
        actual: Any,
        message: str = ""
    ) -> AssertionResult:
        """Record assertion result with traceback information"""
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go up two frames
        line_number = caller_frame.f_lineno
        file_name = caller_frame.f_filename
        
        # Generate detailed traceback
        traceback_info = traceback.format_stack()[-3]  # Get calling frame
        
        result = AssertionResult(
            test_name=self.current_test or "unknown",
            assertion_type=assertion_type,
            passed=passed,
            expected=expected,
            actual=actual,
            message=message,
            traceback_info=traceback_info,
            line_number=line_number,
            file_name=file_name
        )
        
        # Store result
        test_name = self.current_test or "unknown"
        if test_name not in self.results:
            self.results[test_name] = []
        self.results[test_name].append(result)
        
        # Raise exception if assertion failed
        if not passed:
            detailed_message = self._generate_failure_message(result)
            raise AssertionError(detailed_message, result)
            
        return result
        
    def _generate_failure_message(self, result: AssertionResult) -> str:
        """Generate detailed failure message"""
        base_message = f"Assertion failed: {result.message or result.assertion_type.value}"
        
        # Add comparison details
        if result.assertion_type in [ComparisonType.EQUAL, ComparisonType.NOT_EQUAL]:
            diff = self._generate_diff(result.expected, result.actual)
            if diff:
                base_message += f"\n\nDifference:\n{diff}"
                
        base_message += f"\n\nExpected: {repr(result.expected)}"
        base_message += f"\nActual:   {repr(result.actual)}"
        base_message += f"\nFile:     {result.file_name}:{result.line_number}"
        
        return base_message
        
    def _generate_diff(self, expected: Any, actual: Any) -> str:
        """Generate diff for complex objects"""
        try:
            if isinstance(expected, str) and isinstance(actual, str):
                expected_lines = expected.splitlines(keepends=True)
                actual_lines = actual.splitlines(keepends=True)
                
                diff = difflib.unified_diff(
                    expected_lines,
                    actual_lines,
                    fromfile='expected',
                    tofile='actual',
                    lineterm=''
                )
                return ''.join(diff)
            elif isinstance(expected, (dict, list)) and isinstance(actual, (dict, list)):
                expected_str = json.dumps(expected, indent=2, sort_keys=True, default=str)
                actual_str = json.dumps(actual, indent=2, sort_keys=True, default=str)
                return self._generate_diff(expected_str, actual_str)
        except Exception:
            pass
            
        return ""
        
    # Core assertion methods
    def equal(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert equality with detailed comparison"""
        passed = actual == expected
        return self._record_assertion(
            ComparisonType.EQUAL, passed, expected, actual, 
            message or f"Expected {actual} to equal {expected}"
        )
        
    def not_equal(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert inequality"""
        passed = actual != expected
        return self._record_assertion(
            ComparisonType.NOT_EQUAL, passed, expected, actual,
            message or f"Expected {actual} to not equal {expected}"
        )
        
    def is_true(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert value is True"""
        return self.equal(actual, True, message or f"Expected {actual} to be True")
        
    def is_false(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert value is False"""
        return self.equal(actual, False, message or f"Expected {actual} to be False")
        
    def is_none(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert value is None"""
        passed = actual is None
        return self._record_assertion(
            ComparisonType.IS, passed, None, actual,
            message or f"Expected {actual} to be None"
        )
        
    def is_not_none(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert value is not None"""
        passed = actual is not None
        return self._record_assertion(
            ComparisonType.IS_NOT, passed, "not None", actual,
            message or f"Expected {actual} to not be None"
        )
        
    def greater_than(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert actual > expected"""
        passed = actual > expected
        return self._record_assertion(
            ComparisonType.GREATER, passed, expected, actual,
            message or f"Expected {actual} > {expected}"
        )
        
    def greater_than_or_equal(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert actual >= expected"""
        passed = actual >= expected
        return self._record_assertion(
            ComparisonType.GREATER_EQUAL, passed, expected, actual,
            message or f"Expected {actual} >= {expected}"
        )
        
    def less_than(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert actual < expected"""
        passed = actual < expected
        return self._record_assertion(
            ComparisonType.LESS, passed, expected, actual,
            message or f"Expected {actual} < {expected}"
        )
        
    def less_than_or_equal(self, actual: Any, expected: Any, message: str = "") -> AssertionResult:
        """Assert actual <= expected"""
        passed = actual <= expected
        return self._record_assertion(
            ComparisonType.LESS_EQUAL, passed, expected, actual,
            message or f"Expected {actual} <= {expected}"
        )
        
    def contains(self, container: Any, item: Any, message: str = "") -> AssertionResult:
        """Assert container contains item"""
        passed = item in container
        return self._record_assertion(
            ComparisonType.IN, passed, f"{item} in container", container,
            message or f"Expected {container} to contain {item}"
        )
        
    def not_contains(self, container: Any, item: Any, message: str = "") -> AssertionResult:
        """Assert container does not contain item"""
        passed = item not in container
        return self._record_assertion(
            ComparisonType.NOT_IN, passed, f"{item} not in container", container,
            message or f"Expected {container} to not contain {item}"
        )
        
    def isinstance_of(self, actual: Any, expected_type: Type, message: str = "") -> AssertionResult:
        """Assert actual is instance of expected_type"""
        passed = isinstance(actual, expected_type)
        return self._record_assertion(
            ComparisonType.ISINSTANCE, passed, expected_type, type(actual),
            message or f"Expected {actual} to be instance of {expected_type}"
        )
        
    def regex_match(self, actual: str, pattern: str, message: str = "") -> AssertionResult:
        """Assert string matches regex pattern"""
        passed = bool(re.match(pattern, actual))
        return self._record_assertion(
            ComparisonType.REGEX_MATCH, passed, pattern, actual,
            message or f"Expected '{actual}' to match pattern '{pattern}'"
        )
        
    def length_equal(self, actual: Any, expected_length: int, message: str = "") -> AssertionResult:
        """Assert length of object equals expected"""
        actual_length = len(actual)
        return self.equal(
            actual_length, expected_length,
            message or f"Expected length {actual_length} to equal {expected_length}"
        )
        
    def empty(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert object is empty"""
        return self.length_equal(actual, 0, message or f"Expected {actual} to be empty")
        
    def not_empty(self, actual: Any, message: str = "") -> AssertionResult:
        """Assert object is not empty"""
        passed = len(actual) > 0
        return self._record_assertion(
            ComparisonType.GREATER, passed, 0, len(actual),
            message or f"Expected {actual} to not be empty"
        )
        
    # Exception assertions
    def raises(
        self, 
        exception_type: Type[Exception], 
        callable_obj: Callable = None, 
        message: str = ""
    ):
        """Context manager or decorator to assert exception is raised"""
        
        class ExceptionAssertion:
            def __init__(self, framework: AssertionFramework, exc_type: Type[Exception], msg: str):
                self.framework = framework
                self.exception_type = exc_type
                self.message = msg
                self.exception_caught = None
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    # No exception was raised
                    self.framework._record_assertion(
                        ComparisonType.ISINSTANCE, False, self.exception_type, None,
                        self.message or f"Expected {self.exception_type} to be raised"
                    )
                elif issubclass(exc_type, self.exception_type):
                    # Correct exception type
                    self.exception_caught = exc_val
                    self.framework._record_assertion(
                        ComparisonType.ISINSTANCE, True, self.exception_type, exc_type,
                        self.message or f"Successfully caught {self.exception_type}"
                    )
                    return True  # Suppress the exception
                else:
                    # Wrong exception type
                    self.framework._record_assertion(
                        ComparisonType.ISINSTANCE, False, self.exception_type, exc_type,
                        self.message or f"Expected {self.exception_type}, got {exc_type}"
                    )
                    
        if callable_obj:
            # Used as function call
            with ExceptionAssertion(self, exception_type, message) as assertion:
                callable_obj()
            return assertion
        else:
            # Used as context manager
            return ExceptionAssertion(self, exception_type, message)
            
    # Collection assertions
    def all_match(self, items: List[Any], predicate: Callable[[Any], bool], message: str = "") -> AssertionResult:
        """Assert all items match predicate"""
        failed_items = [item for item in items if not predicate(item)]
        passed = len(failed_items) == 0
        return self._record_assertion(
            ComparisonType.EQUAL, passed, [], failed_items,
            message or f"Expected all items to match predicate, failed: {failed_items}"
        )
        
    def any_match(self, items: List[Any], predicate: Callable[[Any], bool], message: str = "") -> AssertionResult:
        """Assert at least one item matches predicate"""
        matching_items = [item for item in items if predicate(item)]
        passed = len(matching_items) > 0
        return self._record_assertion(
            ComparisonType.GREATER, passed, 0, len(matching_items),
            message or f"Expected at least one item to match predicate"
        )
        
    def dict_contains_subset(self, actual: Dict[str, Any], expected_subset: Dict[str, Any], message: str = "") -> AssertionResult:
        """Assert dictionary contains all key-value pairs from subset"""
        missing_keys = []
        wrong_values = []
        
        for key, expected_value in expected_subset.items():
            if key not in actual:
                missing_keys.append(key)
            elif actual[key] != expected_value:
                wrong_values.append((key, actual[key], expected_value))
                
        passed = len(missing_keys) == 0 and len(wrong_values) == 0
        error_details = {
            'missing_keys': missing_keys,
            'wrong_values': wrong_values
        }
        
        return self._record_assertion(
            ComparisonType.CONTAINS, passed, expected_subset, error_details,
            message or f"Dictionary subset assertion failed: {error_details}"
        )
        
    # Fluent interface
    def that(self, value: Any):
        """Start fluent assertion chain"""
        return FluentAssertion(self, value)


class FluentAssertion:
    """Fluent interface for assertions"""
    
    def __init__(self, framework: AssertionFramework, value: Any):
        self.framework = framework
        self.value = value
        
    def equals(self, expected: Any, message: str = ""):
        return self.framework.equal(self.value, expected, message)
        
    def does_not_equal(self, expected: Any, message: str = ""):
        return self.framework.not_equal(self.value, expected, message)
        
    def is_greater_than(self, expected: Any, message: str = ""):
        return self.framework.greater_than(self.value, expected, message)
        
    def is_less_than(self, expected: Any, message: str = ""):
        return self.framework.less_than(self.value, expected, message)
        
    def is_none(self, message: str = ""):
        return self.framework.is_none(self.value, message)
        
    def is_not_none(self, message: str = ""):
        return self.framework.is_not_none(self.value, message)
        
    def contains(self, item: Any, message: str = ""):
        return self.framework.contains(self.value, item, message)
        
    def is_instance_of(self, expected_type: Type, message: str = ""):
        return self.framework.isinstance_of(self.value, expected_type, message)
        
    def has_length(self, expected_length: int, message: str = ""):
        return self.framework.length_equal(self.value, expected_length, message)
        
    def is_empty(self, message: str = ""):
        return self.framework.empty(self.value, message)
        
    def is_not_empty(self, message: str = ""):
        return self.framework.not_empty(self.value, message)