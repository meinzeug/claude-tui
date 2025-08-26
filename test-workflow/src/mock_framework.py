"""Mock Framework with Spy Capabilities - London School TDD

Designed to support London School TDD by:
1. Creating mocks that capture interactions
2. Verifying behavior and collaborations
3. Setting expectations and return values
4. Tracking call order and arguments
"""

import functools
import inspect
from typing import Any, List, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from unittest.mock import Mock as StdMock


@dataclass
class Call:
    """Represents a method call with arguments and metadata"""
    method_name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    return_value: Any = None
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class MockMethod:
    """A mock method that tracks calls and can be configured"""
    
    def __init__(self, name: str, parent_mock=None):
        self.name = name
        self.parent_mock = parent_mock
        self.calls: List[Call] = []
        self._return_value = None
        self._exception_to_raise = None
        self._side_effect = None
        self._call_count = 0
    
    def __call__(self, *args, **kwargs):
        """Execute the mock method call"""
        self._call_count += 1
        
        # Record the call
        call = Call(
            method_name=self.name,
            args=args,
            kwargs=kwargs
        )
        self.calls.append(call)
        
        # Notify parent mock of the call
        if self.parent_mock:
            self.parent_mock._record_call(call)
        
        # Handle side effects
        if self._side_effect:
            if callable(self._side_effect):
                result = self._side_effect(*args, **kwargs)
                call.return_value = result
                return result
            elif isinstance(self._side_effect, list):
                if self._call_count <= len(self._side_effect):
                    result = self._side_effect[self._call_count - 1]
                    if isinstance(result, Exception):
                        call.exception = result
                        raise result
                    call.return_value = result
                    return result
        
        # Handle exception
        if self._exception_to_raise:
            call.exception = self._exception_to_raise
            raise self._exception_to_raise
        
        # Handle return value
        call.return_value = self._return_value
        return self._return_value
    
    def returns(self, value):
        """Set return value for this method"""
        self._return_value = value
        return self
    
    def throws(self, exception):
        """Set exception to raise when called"""
        self._exception_to_raise = exception
        return self
    
    def side_effect(self, effect):
        """Set side effect (function or list of values)"""
        self._side_effect = effect
        return self
    
    @property
    def called(self) -> bool:
        """Check if method was called"""
        return len(self.calls) > 0
    
    @property
    def call_count(self) -> int:
        """Get number of times method was called"""
        return len(self.calls)
    
    def called_with(self, *args, **kwargs) -> bool:
        """Check if method was called with specific arguments"""
        for call in self.calls:
            if call.args == args and call.kwargs == kwargs:
                return True
        return False
    
    def assert_called_once(self):
        """Assert method was called exactly once"""
        if self.call_count != 1:
            raise AssertionError(f"Expected {self.name} to be called once, but was called {self.call_count} times")
    
    def assert_called_with(self, *args, **kwargs):
        """Assert method was called with specific arguments"""
        if not self.called_with(*args, **kwargs):
            raise AssertionError(f"Expected {self.name} to be called with {args}, {kwargs}")


class Mock:
    """London School Mock with spy capabilities
    
    Focuses on:
    1. Recording all interactions
    2. Allowing behavior verification
    3. Supporting method expectations
    4. Tracking call order
    """
    
    def __init__(self, spec=None, name="Mock"):
        self._name = name
        self._spec = spec
        self._methods: Dict[str, MockMethod] = {}
        self._all_calls: List[Call] = []
        self._call_order: List[str] = []
    
    def __getattr__(self, name: str) -> MockMethod:
        """Create mock methods on demand"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name not in self._methods:
            self._methods[name] = MockMethod(name, parent_mock=self)
        
        return self._methods[name]
    
    def _record_call(self, call: Call) -> None:
        """Record a call for order tracking"""
        self._all_calls.append(call)
        self._call_order.append(call.method_name)
    
    def reset(self) -> None:
        """Reset all call history and expectations"""
        for method in self._methods.values():
            method.calls.clear()
            method._call_count = 0
        self._all_calls.clear()
        self._call_order.clear()
    
    @property
    def call_order(self) -> List[str]:
        """Get the order in which methods were called"""
        return self._call_order.copy()
    
    @property
    def all_calls(self) -> List[Call]:
        """Get all calls made to this mock"""
        return self._all_calls.copy()
    
    def __repr__(self):
        return f"<Mock name='{self._name}'>"


def create_mock(spec=None, name="Mock") -> Mock:
    """Factory function to create a mock object"""
    return Mock(spec=spec, name=name)


class PartialMock:
    """Partial mock that wraps a real object
    
    Allows mocking specific methods while keeping others real.
    """
    
    def __init__(self, real_object):
        self._real_object = real_object
        self._mocked_methods: Dict[str, MockMethod] = {}
        self._original_methods: Dict[str, Callable] = {}
    
    def __getattr__(self, name: str):
        if name in self._mocked_methods:
            return self._mocked_methods[name]
        return getattr(self._real_object, name)
    
    def mock_method(self, method_name: str) -> MockMethod:
        """Mock a specific method on the real object"""
        if method_name not in self._mocked_methods:
            # Store original method
            if hasattr(self._real_object, method_name):
                self._original_methods[method_name] = getattr(self._real_object, method_name)
            
            # Create mock method
            mock_method = MockMethod(method_name)
            self._mocked_methods[method_name] = mock_method
            
            # Replace method on real object
            setattr(self._real_object, method_name, mock_method)
        
        return self._mocked_methods[method_name]
    
    def restore_method(self, method_name: str) -> None:
        """Restore original method"""
        if method_name in self._original_methods:
            setattr(self._real_object, method_name, self._original_methods[method_name])
            del self._original_methods[method_name]
            del self._mocked_methods[method_name]
    
    def restore_all(self) -> None:
        """Restore all mocked methods"""
        for method_name in list(self._mocked_methods.keys()):
            self.restore_method(method_name)


def create_partial_mock(real_object) -> PartialMock:
    """Create a partial mock of a real object"""
    return PartialMock(real_object)


def verify_call_order(mock_obj: Mock, expected_order: List[str]) -> None:
    """Verify that methods were called in the expected order"""
    actual_order = mock_obj.call_order
    
    # Check if expected calls appear in order (allowing for extra calls)
    expected_index = 0
    for actual_call in actual_order:
        if expected_index < len(expected_order) and actual_call == expected_order[expected_index]:
            expected_index += 1
    
    if expected_index < len(expected_order):
        raise AssertionError(
            f"Expected call order not satisfied. Expected: {expected_order}, Actual: {actual_order}"
        )


def spy_on(obj, method_name: str) -> MockMethod:
    """Create a spy on a real object's method
    
    The method still executes normally but calls are recorded.
    """
    original_method = getattr(obj, method_name)
    spy_method = MockMethod(method_name)
    
    def spy_wrapper(*args, **kwargs):
        # Record the call
        call = Call(
            method_name=method_name,
            args=args,
            kwargs=kwargs
        )
        spy_method.calls.append(call)
        
        try:
            # Execute original method
            result = original_method(*args, **kwargs)
            call.return_value = result
            return result
        except Exception as e:
            call.exception = e
            raise
    
    # Replace method with spy wrapper
    setattr(obj, method_name, spy_wrapper)
    
    # Store original for potential restoration
    spy_method._original_method = original_method
    spy_method._target_object = obj
    
    return spy_method


def restore_spy(spy_method: MockMethod) -> None:
    """Restore original method from spy"""
    if hasattr(spy_method, '_original_method') and hasattr(spy_method, '_target_object'):
        setattr(spy_method._target_object, spy_method.name, spy_method._original_method)


# Context manager for temporary mocking
class MockContext:
    """Context manager for temporary mocks
    
    Ensures mocks are cleaned up after use.
    """
    
    def __init__(self):
        self._mocks: List[Mock] = []
        self._spies: List[MockMethod] = []
        self._partial_mocks: List[PartialMock] = []
    
    def mock(self, spec=None, name="Mock") -> Mock:
        """Create a mock within this context"""
        mock_obj = create_mock(spec=spec, name=name)
        self._mocks.append(mock_obj)
        return mock_obj
    
    def spy(self, obj, method_name: str) -> MockMethod:
        """Create a spy within this context"""
        spy_method = spy_on(obj, method_name)
        self._spies.append(spy_method)
        return spy_method
    
    def partial_mock(self, real_object) -> PartialMock:
        """Create a partial mock within this context"""
        partial = create_partial_mock(real_object)
        self._partial_mocks.append(partial)
        return partial
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up all mocks
        for mock_obj in self._mocks:
            mock_obj.reset()
        
        # Restore all spies
        for spy_method in self._spies:
            restore_spy(spy_method)
        
        # Restore all partial mocks
        for partial in self._partial_mocks:
            partial.restore_all()
        
        # Clear references
        self._mocks.clear()
        self._spies.clear()
        self._partial_mocks.clear()


# Convenience function for creating mock contexts
def mock_context() -> MockContext:
    """Create a mock context for temporary mocking"""
    return MockContext()