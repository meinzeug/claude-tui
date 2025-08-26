"""Chainable Assertion Library - London School TDD Support

Designed to support behavior verification:
1. Fluent, readable assertion syntax
2. Clear error messages for test failures
3. Async assertion support
4. Deep object comparison
5. Type checking and property validation
"""

import asyncio
import inspect
from typing import Any, Union, Callable, Awaitable, List, Dict, Optional
from dataclasses import dataclass
from collections.abc import Iterable, Mapping


class AssertionError(Exception):
    """Custom assertion error with detailed messaging"""
    
    def __init__(self, message: str, actual=None, expected=None, operator=None):
        super().__init__(message)
        self.message = message
        self.actual = actual
        self.expected = expected
        self.operator = operator
    
    def __str__(self):
        if self.actual is not None and self.expected is not None:
            return f"{self.message}\n  actual: {repr(self.actual)}\n  expected: {repr(self.expected)}"
        return self.message


@dataclass
class AssertionContext:
    """Context for building assertion chains"""
    actual: Any
    negated: bool = False
    async_mode: bool = False
    message_prefix: str = ""


class AsyncAssertionChain:
    """Async assertion chain for handling promises/coroutines"""
    
    def __init__(self, context: AssertionContext):
        self._context = context
    
    async def equal(self, expected: Any) -> 'AssertionChain':
        """Assert async result equals expected value"""
        if asyncio.iscoroutine(self._context.actual):
            actual = await self._context.actual
        else:
            actual = self._context.actual
        
        new_context = AssertionContext(actual, self._context.negated)
        chain = AssertionChain(new_context)
        return chain.equal(expected)
    
    async def resolve(self) -> 'AssertionChain':
        """Resolve the async value and return sync chain"""
        if asyncio.iscoroutine(self._context.actual):
            actual = await self._context.actual
        else:
            actual = self._context.actual
        
        new_context = AssertionContext(actual, self._context.negated)
        return AssertionChain(new_context)
    
    async def reject(self, expected_error_type=None) -> 'AssertionChain':
        """Assert that async operation raises an error"""
        exception_raised = None
        
        try:
            if asyncio.iscoroutine(self._context.actual):
                await self._context.actual
            else:
                self._context.actual
        except AssertionError:
            # Re-raise AssertionError from our own assertions
            raise
        except Exception as e:
            exception_raised = e
        
        # Check if exception was raised as expected
        if not exception_raised:
            if not self._context.negated:
                raise AssertionError("Expected operation to reject, but it resolved")
        else:
            if self._context.negated:
                raise AssertionError(f"Expected operation not to reject, but it rejected with: {exception_raised}")
            
            if expected_error_type and not isinstance(exception_raised, expected_error_type):
                raise AssertionError(
                    f"Expected rejection with {expected_error_type.__name__}, got {type(exception_raised).__name__}: {exception_raised}"
                )
        
        return AssertionChain(AssertionContext(None, False))


class AssertionChain:
    """Fluent assertion chain supporting method chaining"""
    
    def __init__(self, context: AssertionContext):
        self._context = context
    
    # Chaining modifiers
    @property
    def to(self) -> 'AssertionChain':
        """Grammatical sugar for fluent interface"""
        return self
    
    @property
    def be(self) -> 'AssertionChain':
        """Grammatical sugar for fluent interface"""
        return self
    
    @property
    def been(self) -> 'AssertionChain':
        """Grammatical sugar for fluent interface"""
        return self
    
    @property
    def have(self) -> 'AssertionChain':
        """Grammatical sugar for fluent interface"""
        return self
    
    @property
    def and_(self) -> 'AssertionChain':
        """Continue chain with 'and'"""
        return self
    
    @property
    def and_be(self) -> 'AssertionChain':
        """Continue chain with 'and be'"""
        return self
    
    def and_contain(self, item: Any) -> 'AssertionChain':
        """Continue chain with contain assertion"""
        return self.contain(item)
    
    @property
    def not_(self) -> 'AssertionChain':
        """Negate the next assertion"""
        new_context = AssertionContext(
            self._context.actual, 
            not self._context.negated,
            self._context.async_mode,
            self._context.message_prefix
        )
        return AssertionChain(new_context)
    
    @property
    def eventually(self) -> AsyncAssertionChain:
        """Switch to async assertion mode"""
        new_context = AssertionContext(
            self._context.actual,
            self._context.negated,
            True,
            self._context.message_prefix
        )
        return AsyncAssertionChain(new_context)
    
    # Core assertions
    def equal(self, expected: Any) -> 'AssertionChain':
        """Assert equality"""
        actual = self._context.actual
        
        if self._context.negated:
            if actual == expected:
                raise AssertionError(
                    f"Expected {actual} not to equal {expected}",
                    actual=actual,
                    expected=expected,
                    operator="!="
                )
        else:
            if actual != expected:
                raise AssertionError(
                    f"Expected {actual} to equal {expected}",
                    actual=actual,
                    expected=expected,
                    operator="=="
                )
        
        return self
    
    def deep_equal(self, expected: Any) -> 'AssertionChain':
        """Assert deep equality for complex objects"""
        actual = self._context.actual
        
        def deep_equals(a, b) -> bool:
            if type(a) != type(b):
                return False
            
            if isinstance(a, (list, tuple)):
                if len(a) != len(b):
                    return False
                return all(deep_equals(x, y) for x, y in zip(a, b))
            
            if isinstance(a, dict):
                if set(a.keys()) != set(b.keys()):
                    return False
                return all(deep_equals(a[k], b[k]) for k in a.keys())
            
            return a == b
        
        is_equal = deep_equals(actual, expected)
        
        if self._context.negated:
            if is_equal:
                raise AssertionError(
                    f"Expected {actual} not to deep equal {expected}",
                    actual=actual,
                    expected=expected,
                    operator="deep !="
                )
        else:
            if not is_equal:
                raise AssertionError(
                    f"Expected {actual} to deep equal {expected}",
                    actual=actual,
                    expected=expected,
                    operator="deep =="
                )
        
        return self
    
    def a(self, type_or_name: Union[type, str]) -> 'AssertionChain':
        """Assert type"""
        actual = self._context.actual
        
        if isinstance(type_or_name, str):
            actual_type_name = type(actual).__name__
            expected_type_name = type_or_name
            matches = actual_type_name == expected_type_name
        else:
            matches = isinstance(actual, type_or_name)
            expected_type_name = type_or_name.__name__
            actual_type_name = type(actual).__name__
        
        if self._context.negated:
            if matches:
                raise AssertionError(
                    f"Expected {actual} not to be a {expected_type_name}",
                    actual=actual_type_name,
                    expected=f"not {expected_type_name}"
                )
        else:
            if not matches:
                raise AssertionError(
                    f"Expected {actual} to be a {expected_type_name}, but got {actual_type_name}",
                    actual=actual_type_name,
                    expected=expected_type_name
                )
        
        return self
    
    def greater_than(self, value: Union[int, float]) -> 'AssertionChain':
        """Assert greater than"""
        actual = self._context.actual
        
        if self._context.negated:
            if actual > value:
                raise AssertionError(
                    f"Expected {actual} not to be greater than {value}",
                    actual=actual,
                    expected=f"<= {value}"
                )
        else:
            if actual <= value:
                raise AssertionError(
                    f"Expected {actual} to be greater than {value}",
                    actual=actual,
                    expected=f"> {value}"
                )
        
        return self
    
    def less_than(self, value: Union[int, float]) -> 'AssertionChain':
        """Assert less than"""
        actual = self._context.actual
        
        if self._context.negated:
            if actual < value:
                raise AssertionError(
                    f"Expected {actual} not to be less than {value}",
                    actual=actual,
                    expected=f">= {value}"
                )
        else:
            if actual >= value:
                raise AssertionError(
                    f"Expected {actual} to be less than {value}",
                    actual=actual,
                    expected=f"< {value}"
                )
        
        return self
    
    def contain(self, item: Any) -> 'AssertionChain':
        """Assert contains item"""
        actual = self._context.actual
        
        try:
            contains = item in actual
        except TypeError:
            contains = False
        
        if self._context.negated:
            if contains:
                raise AssertionError(
                    f"Expected {actual} not to contain {item}",
                    actual=actual,
                    expected=f"not containing {item}"
                )
        else:
            if not contains:
                raise AssertionError(
                    f"Expected {actual} to contain {item}",
                    actual=actual,
                    expected=f"containing {item}"
                )
        
        return self
    
    def length(self, expected_length: int) -> 'AssertionChain':
        """Assert length/size"""
        actual = self._context.actual
        
        try:
            actual_length = len(actual)
        except TypeError:
            raise AssertionError(
                f"Expected {actual} to have length, but it has no length",
                actual=type(actual).__name__,
                expected="object with length"
            )
        
        if self._context.negated:
            if actual_length == expected_length:
                raise AssertionError(
                    f"Expected {actual} not to have length {expected_length}",
                    actual=actual_length,
                    expected=f"!= {expected_length}"
                )
        else:
            if actual_length != expected_length:
                raise AssertionError(
                    f"Expected {actual} to have length {expected_length}, but got {actual_length}",
                    actual=actual_length,
                    expected=expected_length
                )
        
        return self
    
    def property(self, prop_name: str) -> 'PropertyAssertionChain':
        """Assert object has property"""
        actual = self._context.actual
        
        # Check for property differently based on object type
        if isinstance(actual, dict):
            has_property = prop_name in actual
            prop_value = actual.get(prop_name) if has_property else None
        else:
            has_property = hasattr(actual, prop_name)
            prop_value = getattr(actual, prop_name, None) if has_property else None
        
        if self._context.negated:
            if has_property:
                raise AssertionError(
                    f"Expected {actual} not to have property '{prop_name}'",
                    actual=f"has {prop_name}",
                    expected=f"no {prop_name}"
                )
            # Return chain with None value since property doesn't exist
            return PropertyAssertionChain(AssertionContext(None, False), prop_name)
        else:
            if not has_property:
                raise AssertionError(
                    f"Expected {actual} to have property '{prop_name}'",
                    actual=f"no {prop_name}",
                    expected=f"has {prop_name}"
                )
            
            return PropertyAssertionChain(AssertionContext(prop_value, False), prop_name)
    
    def throws(self, expected_exception_type=None, message_pattern=None) -> 'AssertionChain':
        """Assert that calling actual throws an exception"""
        actual = self._context.actual
        
        if not callable(actual):
            raise AssertionError(
                f"Expected {actual} to be callable for throws assertion",
                actual=type(actual).__name__,
                expected="callable"
            )
        
        exception_raised = None
        try:
            result = actual()
            if asyncio.iscoroutine(result):
                # Handle async functions that weren't awaited
                result.close()  # Clean up the coroutine
                raise AssertionError(
                    "Async function used with throws() - use 'await expect(fn()).to.eventually.reject()' instead"
                )
        except Exception as e:
            exception_raised = e
        
        if self._context.negated:
            if exception_raised:
                raise AssertionError(
                    f"Expected {actual} not to throw, but it threw: {exception_raised}",
                    actual=str(exception_raised),
                    expected="no exception"
                )
        else:
            if not exception_raised:
                raise AssertionError(
                    f"Expected {actual} to throw an exception, but it didn't",
                    actual="no exception",
                    expected="exception"
                )
            
            if expected_exception_type and not isinstance(exception_raised, expected_exception_type):
                raise AssertionError(
                    f"Expected {actual} to throw {expected_exception_type.__name__}, but got {type(exception_raised).__name__}: {exception_raised}",
                    actual=type(exception_raised).__name__,
                    expected=expected_exception_type.__name__
                )
            
            if message_pattern and message_pattern not in str(exception_raised):
                raise AssertionError(
                    f"Expected exception message to contain '{message_pattern}', but got: {exception_raised}",
                    actual=str(exception_raised),
                    expected=f"containing '{message_pattern}'"
                )
        
        return self


class PropertyAssertionChain(AssertionChain):
    """Specialized assertion chain for property assertions"""
    
    def __init__(self, context: AssertionContext, property_name: str):
        super().__init__(context)
        self._property_name = property_name
    
    def with_value(self, expected_value: Any) -> AssertionChain:
        """Assert property has specific value"""
        return self.equal(expected_value)
    
    def of_type(self, expected_type: Union[type, str]) -> AssertionChain:
        """Assert property is of specific type"""
        return self.a(expected_type)


def expect(actual: Any) -> AssertionChain:
    """Create an assertion chain for the given value
    
    Usage:
        expect(42).to.equal(42)
        expect([1, 2, 3]).to.have.length(3)
        expect({'key': 'value'}).to.have.property('key').with_value('value')
        await expect(async_function()).to.eventually.equal('result')
    """
    context = AssertionContext(actual)
    return AssertionChain(context)


# Convenience functions for common assertions
def assert_equal(actual: Any, expected: Any, message: str = "") -> None:
    """Simple equality assertion"""
    try:
        expect(actual).to.equal(expected)
    except AssertionError as e:
        if message:
            raise AssertionError(f"{message}: {e.message}", actual=actual, expected=expected)
        raise


def assert_deep_equal(actual: Any, expected: Any, message: str = "") -> None:
    """Simple deep equality assertion"""
    try:
        expect(actual).to.deep_equal(expected)
    except AssertionError as e:
        if message:
            raise AssertionError(f"{message}: {e.message}", actual=actual, expected=expected)
        raise


def assert_throws(fn: Callable, exception_type=None, message: str = "") -> None:
    """Simple throws assertion"""
    try:
        expect(fn).to.throws(exception_type)
    except AssertionError as e:
        if message:
            raise AssertionError(f"{message}: {e.message}")
        raise


async def assert_eventually_equal(awaitable: Awaitable[Any], expected: Any, message: str = "") -> None:
    """Simple async equality assertion"""
    try:
        await expect(awaitable).to.eventually.equal(expected)
    except AssertionError as e:
        if message:
            raise AssertionError(f"{message}: {e.message}")
        raise
