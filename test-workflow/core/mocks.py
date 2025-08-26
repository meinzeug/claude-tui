"""
Mock Framework - Comprehensive mocking system for test-workflow
Integrates with test runner and provides advanced mocking capabilities
"""

import inspect
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import contextlib


class MockType(Enum):
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    CLASS = "class"
    MODULE = "module"
    ATTRIBUTE = "attribute"


@dataclass
class MockCall:
    """Record of a mock call with detailed information"""
    mock_name: str
    call_time: float
    args: tuple
    kwargs: dict
    return_value: Any = None
    exception: Optional[Exception] = None
    call_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mock_name': self.mock_name,
            'call_time': self.call_time,
            'args': [str(arg) for arg in self.args],
            'kwargs': {k: str(v) for k, v in self.kwargs.items()},
            'return_value': str(self.return_value),
            'exception': str(self.exception) if self.exception else None,
            'call_count': self.call_count
        }


@dataclass
class MockConfiguration:
    """Configuration for a mock object"""
    name: str
    mock_type: MockType
    return_value: Any = None
    side_effect: Any = None
    spec: Any = None
    autospec: bool = False
    return_value_sequence: List[Any] = field(default_factory=list)
    call_count_limit: Optional[int] = None
    delay_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'mock_type': self.mock_type.value,
            'return_value': str(self.return_value),
            'side_effect': str(self.side_effect) if self.side_effect else None,
            'spec': str(self.spec) if self.spec else None,
            'autospec': self.autospec,
            'return_value_sequence': [str(v) for v in self.return_value_sequence],
            'call_count_limit': self.call_count_limit,
            'delay_seconds': self.delay_seconds
        }


class MockFramework:
    """
    Comprehensive mock framework that integrates with test runner
    Provides advanced mocking, stubbing, and spying capabilities
    """
    
    def __init__(self):
        self.mocks: Dict[str, Dict[str, Any]] = {}  # test_name -> mock_name -> mock_data
        self.call_history: Dict[str, List[MockCall]] = {}  # test_name -> calls
        self.configurations: Dict[str, Dict[str, MockConfiguration]] = {}  # test_name -> mock configurations
        self.active_patches: Dict[str, List[Any]] = {}  # test_name -> active patches
        self.current_test: Optional[str] = None
        
    def reset_for_test(self, test_name: str) -> None:
        """Reset mock framework for new test"""
        self.current_test = test_name
        
        # Clean up previous test patches
        if test_name in self.active_patches:
            for patch_obj in self.active_patches[test_name]:
                try:
                    patch_obj.stop()
                except Exception:
                    pass
                    
        # Initialize containers for this test
        self.mocks[test_name] = {}
        self.call_history[test_name] = []
        self.configurations[test_name] = {}
        self.active_patches[test_name] = []
        
    def get_used_mocks(self, test_name: str) -> List[str]:
        """Get list of mock names used in test"""
        return list(self.mocks.get(test_name, {}).keys())
        
    def get_call_history(self, test_name: str) -> List[MockCall]:
        """Get call history for test"""
        return self.call_history.get(test_name, [])
        
    def _record_call(
        self,
        mock_name: str,
        args: tuple,
        kwargs: dict,
        return_value: Any = None,
        exception: Exception = None
    ) -> None:
        """Record a mock call"""
        test_name = self.current_test or "unknown"
        
        call = MockCall(
            mock_name=mock_name,
            call_time=time.time(),
            args=args,
            kwargs=kwargs,
            return_value=return_value,
            exception=exception
        )
        
        if test_name not in self.call_history:
            self.call_history[test_name] = []
        self.call_history[test_name].append(call)
        
    def create_mock(
        self,
        name: str,
        spec: Any = None,
        autospec: bool = False,
        **kwargs
    ) -> Mock:
        """Create a mock object with enhanced tracking"""
        test_name = self.current_test or "unknown"
        
        # Create configuration
        config = MockConfiguration(
            name=name,
            mock_type=MockType.FUNCTION,
            spec=spec,
            autospec=autospec,
            **kwargs
        )
        
        if test_name not in self.configurations:
            self.configurations[test_name] = {}
        self.configurations[test_name][name] = config
        
        # Create mock with tracking
        if autospec and spec:
            mock_obj = Mock(spec=spec)
        elif spec:
            mock_obj = Mock(spec=spec)
        else:
            mock_obj = Mock()
            
        # Wrap the mock to record calls
        original_call = mock_obj.__call__
        
        def tracking_call(*args, **kwargs):
            try:
                # Apply delay if configured
                if config.delay_seconds > 0:
                    time.sleep(config.delay_seconds)
                    
                # Check call count limit
                if config.call_count_limit is not None:
                    current_calls = len([c for c in self.call_history.get(test_name, []) if c.mock_name == name])
                    if current_calls >= config.call_count_limit:
                        raise Exception(f"Mock {name} exceeded call count limit of {config.call_count_limit}")
                        
                # Handle return value sequence
                if config.return_value_sequence:
                    current_calls = len([c for c in self.call_history.get(test_name, []) if c.mock_name == name])
                    if current_calls < len(config.return_value_sequence):
                        result = config.return_value_sequence[current_calls]
                        self._record_call(name, args, kwargs, result)
                        return result
                        
                # Normal mock behavior
                result = original_call(*args, **kwargs)
                self._record_call(name, args, kwargs, result)
                return result
                
            except Exception as e:
                self._record_call(name, args, kwargs, exception=e)
                raise
                
        mock_obj.__call__ = tracking_call
        
        # Store mock
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][name] = mock_obj
        
        return mock_obj
        
    def create_spy(self, target: Any, method_name: str) -> Mock:
        """Create a spy that tracks calls while preserving original behavior"""
        test_name = self.current_test or "unknown"
        spy_name = f"spy_{target.__class__.__name__}_{method_name}"
        
        original_method = getattr(target, method_name)
        
        def spy_wrapper(*args, **kwargs):
            try:
                result = original_method(*args, **kwargs)
                self._record_call(spy_name, args, kwargs, result)
                return result
            except Exception as e:
                self._record_call(spy_name, args, kwargs, exception=e)
                raise
                
        # Replace method with spy
        setattr(target, method_name, spy_wrapper)
        
        # Create mock for assertions
        spy_mock = Mock(wraps=spy_wrapper)
        
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][spy_name] = spy_mock
        
        return spy_mock
        
    def stub_method(
        self,
        target: Any,
        method_name: str,
        return_value: Any = None,
        side_effect: Any = None
    ) -> Mock:
        """Stub a method with specified behavior"""
        test_name = self.current_test or "unknown"
        stub_name = f"stub_{target.__class__.__name__}_{method_name}"
        
        stub_mock = Mock(return_value=return_value, side_effect=side_effect)
        
        def stub_wrapper(*args, **kwargs):
            try:
                result = stub_mock(*args, **kwargs)
                self._record_call(stub_name, args, kwargs, result)
                return result
            except Exception as e:
                self._record_call(stub_name, args, kwargs, exception=e)
                raise
                
        # Replace method with stub
        setattr(target, method_name, stub_wrapper)
        
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][stub_name] = stub_mock
        
        return stub_mock
        
    def patch_object(
        self,
        target: Any,
        attribute: str,
        new: Any = None,
        spec: Any = None,
        autospec: bool = False
    ) -> Mock:
        """Patch an object attribute with tracking"""
        test_name = self.current_test or "unknown"
        patch_name = f"patch_{target.__class__.__name__}_{attribute}"
        
        patcher = patch.object(target, attribute, new=new, spec=spec, autospec=autospec)
        mock_obj = patcher.start()
        
        # Track the patcher for cleanup
        if test_name not in self.active_patches:
            self.active_patches[test_name] = []
        self.active_patches[test_name].append(patcher)
        
        # Add call tracking if it's a mock
        if isinstance(mock_obj, Mock):
            original_call = mock_obj.__call__
            
            def tracking_call(*args, **kwargs):
                try:
                    result = original_call(*args, **kwargs)
                    self._record_call(patch_name, args, kwargs, result)
                    return result
                except Exception as e:
                    self._record_call(patch_name, args, kwargs, exception=e)
                    raise
                    
            mock_obj.__call__ = tracking_call
            
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][patch_name] = mock_obj
        
        return mock_obj
        
    def patch_module(
        self,
        module_path: str,
        new: Any = None,
        spec: Any = None
    ) -> Mock:
        """Patch a module with tracking"""
        test_name = self.current_test or "unknown"
        patch_name = f"patch_module_{module_path.replace('.', '_')}"
        
        patcher = patch(module_path, new=new, spec=spec)
        mock_obj = patcher.start()
        
        # Track the patcher for cleanup
        if test_name not in self.active_patches:
            self.active_patches[test_name] = []
        self.active_patches[test_name].append(patcher)
        
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][patch_name] = mock_obj
        
        return mock_obj
        
    @contextlib.contextmanager
    def temporary_patch(
        self,
        target: str,
        new: Any = None,
        **kwargs
    ):
        """Context manager for temporary patches"""
        test_name = self.current_test or "unknown"
        patch_name = f"temp_patch_{target.replace('.', '_')}"
        
        with patch(target, new=new, **kwargs) as mock_obj:
            if test_name not in self.mocks:
                self.mocks[test_name] = {}
            self.mocks[test_name][patch_name] = mock_obj
            yield mock_obj
            
    def create_async_mock(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Any = None
    ) -> Mock:
        """Create an async mock for coroutine testing"""
        import asyncio
        
        async_mock = Mock()
        
        if side_effect:
            async def async_side_effect(*args, **kwargs):
                if inspect.iscoroutinefunction(side_effect):
                    return await side_effect(*args, **kwargs)
                else:
                    return side_effect(*args, **kwargs)
            async_mock.return_value = asyncio.coroutine(async_side_effect)()
        else:
            async def async_return(*args, **kwargs):
                return return_value
            async_mock.return_value = asyncio.coroutine(async_return)()
            
        test_name = self.current_test or "unknown"
        if test_name not in self.mocks:
            self.mocks[test_name] = {}
        self.mocks[test_name][name] = async_mock
        
        return async_mock
        
    # Assertion helpers for mocks
    def assert_called(self, mock_name: str) -> None:
        """Assert mock was called at least once"""
        calls = self._get_calls_for_mock(mock_name)
        if not calls:
            raise AssertionError(f"Mock {mock_name} was never called")
            
    def assert_called_once(self, mock_name: str) -> None:
        """Assert mock was called exactly once"""
        calls = self._get_calls_for_mock(mock_name)
        if len(calls) != 1:
            raise AssertionError(f"Mock {mock_name} was called {len(calls)} times, expected 1")
            
    def assert_called_with(
        self,
        mock_name: str,
        *expected_args,
        **expected_kwargs
    ) -> None:
        """Assert mock was called with specific arguments"""
        calls = self._get_calls_for_mock(mock_name)
        if not calls:
            raise AssertionError(f"Mock {mock_name} was never called")
            
        last_call = calls[-1]
        if last_call.args != expected_args or last_call.kwargs != expected_kwargs:
            raise AssertionError(
                f"Mock {mock_name} called with args={last_call.args}, kwargs={last_call.kwargs}, "
                f"expected args={expected_args}, kwargs={expected_kwargs}"
            )
            
    def assert_not_called(self, mock_name: str) -> None:
        """Assert mock was never called"""
        calls = self._get_calls_for_mock(mock_name)
        if calls:
            raise AssertionError(f"Mock {mock_name} was called {len(calls)} times, expected 0")
            
    def assert_call_count(self, mock_name: str, expected_count: int) -> None:
        """Assert mock was called specific number of times"""
        calls = self._get_calls_for_mock(mock_name)
        if len(calls) != expected_count:
            raise AssertionError(
                f"Mock {mock_name} was called {len(calls)} times, expected {expected_count}"
            )
            
    def _get_calls_for_mock(self, mock_name: str) -> List[MockCall]:
        """Get all calls for a specific mock"""
        test_name = self.current_test or "unknown"
        all_calls = self.call_history.get(test_name, [])
        return [call for call in all_calls if call.mock_name == mock_name]
        
    def get_mock_statistics(self, test_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics about mocks used in test"""
        mocks = self.mocks.get(test_name, {})
        calls = self.call_history.get(test_name, [])
        configs = self.configurations.get(test_name, {})
        
        call_counts = {}
        for call in calls:
            call_counts[call.mock_name] = call_counts.get(call.mock_name, 0) + 1
            
        return {
            'total_mocks': len(mocks),
            'total_calls': len(calls),
            'mock_names': list(mocks.keys()),
            'call_counts': call_counts,
            'configurations': {name: config.to_dict() for name, config in configs.items()},
            'call_history': [call.to_dict() for call in calls]
        }
        
    # Factory methods for common mock patterns
    def create_database_mock(self) -> Mock:
        """Create a mock database connection"""
        db_mock = self.create_mock("database")
        db_mock.execute.return_value = Mock()
        db_mock.fetchone.return_value = None
        db_mock.fetchall.return_value = []
        db_mock.commit.return_value = None
        db_mock.rollback.return_value = None
        return db_mock
        
    def create_http_client_mock(self, status_code: int = 200, response_data: dict = None) -> Mock:
        """Create a mock HTTP client"""
        http_mock = self.create_mock("http_client")
        
        response_mock = Mock()
        response_mock.status_code = status_code
        response_mock.json.return_value = response_data or {}
        response_mock.text = json.dumps(response_data or {})
        
        http_mock.get.return_value = response_mock
        http_mock.post.return_value = response_mock
        http_mock.put.return_value = response_mock
        http_mock.delete.return_value = response_mock
        
        return http_mock
        
    def create_file_system_mock(self) -> Mock:
        """Create a mock file system"""
        fs_mock = self.create_mock("file_system")
        fs_mock.exists.return_value = True
        fs_mock.read.return_value = ""
        fs_mock.write.return_value = None
        fs_mock.delete.return_value = None
        fs_mock.list_files.return_value = []
        return fs_mock