"""Unit Tests for Mock Framework - London School TDD

Tests the spy capabilities and interaction verification features
of the mock framework.
"""

import pytest
from src.mock_framework import (
    Mock, MockMethod, create_mock, create_partial_mock,
    verify_call_order, spy_on, restore_spy, mock_context
)


class TestMockBehavior:
    """Test basic mock behavior and spy capabilities"""
    
    def test_should_create_mock_methods_on_demand(self):
        """Mock should create methods dynamically when accessed"""
        # Act
        mock = create_mock()
        method = mock.some_method
        
        # Assert
        assert isinstance(method, MockMethod)
        assert method.name == "some_method"
    
    def test_should_track_method_calls(self):
        """Mock should track all method calls"""
        # Arrange
        mock = create_mock()
        
        # Act
        mock.test_method('arg1', 'arg2')
        mock.test_method('arg3')
        
        # Assert
        assert mock.test_method.called
        assert mock.test_method.call_count == 2
        assert len(mock.test_method.calls) == 2
        
        # Verify call details
        assert mock.test_method.calls[0].args == ('arg1', 'arg2')
        assert mock.test_method.calls[1].args == ('arg3',)
    
    def test_should_support_return_value_configuration(self):
        """Mock should allow setting return values"""
        # Arrange
        mock = create_mock()
        mock.get_value.returns(42)
        
        # Act
        result = mock.get_value()
        
        # Assert
        assert result == 42
        assert mock.get_value.called
    
    def test_should_support_exception_configuration(self):
        """Mock should allow setting exceptions to throw"""
        # Arrange
        mock = create_mock()
        mock.failing_method.throws(ValueError("Test error"))
        
        # Act & Assert
        with pytest.raises(ValueError, match="Test error"):
            mock.failing_method()
        
        assert mock.failing_method.called
    
    def test_should_support_side_effects(self):
        """Mock should support side effects"""
        # Arrange
        mock = create_mock()
        
        def side_effect_fn(arg):
            return f"processed_{arg}"
        
        mock.process.side_effect(side_effect_fn)
        
        # Act
        result = mock.process("test")
        
        # Assert
        assert result == "processed_test"
        assert mock.process.called
    
    def test_should_support_side_effect_list(self):
        """Mock should support list of side effects"""
        # Arrange
        mock = create_mock()
        mock.get_next.side_effect([1, 2, ValueError("No more values")])
        
        # Act & Assert
        assert mock.get_next() == 1
        assert mock.get_next() == 2
        
        with pytest.raises(ValueError, match="No more values"):
            mock.get_next()
    
    def test_should_verify_call_arguments(self):
        """Mock should verify specific call arguments"""
        # Arrange
        mock = create_mock()
        mock.test_method('expected_arg', key='expected_value')
        
        # Act & Assert
        assert mock.test_method.called_with('expected_arg', key='expected_value')
        assert not mock.test_method.called_with('wrong_arg')
    
    def test_should_provide_assertion_helpers(self):
        """Mock should provide convenient assertion methods"""
        # Arrange
        mock = create_mock()
        mock.single_call_method()
        
        # Act & Assert - Should not raise
        mock.single_call_method.assert_called_once()
        
        # Should raise for wrong arguments
        with pytest.raises(AssertionError):
            mock.single_call_method.assert_called_with('wrong_arg')
    
    def test_should_track_call_order(self):
        """Mock should track the order of method calls"""
        # Arrange
        mock = create_mock()
        
        # Act
        mock.first_method()
        mock.second_method()
        mock.third_method()
        
        # Assert
        assert mock.call_order == ['first_method', 'second_method', 'third_method']
    
    def test_should_reset_call_history(self):
        """Mock should be able to reset its call history"""
        # Arrange
        mock = create_mock()
        mock.test_method()
        assert mock.test_method.called
        
        # Act
        mock.reset()
        
        # Assert
        assert not mock.test_method.called
        assert mock.test_method.call_count == 0
        assert len(mock.call_order) == 0


class TestCallOrderVerification:
    """Test call order verification functionality"""
    
    def test_should_verify_correct_call_order(self):
        """Should verify that methods were called in expected order"""
        # Arrange
        mock = create_mock()
        mock.step1()
        mock.step2()
        mock.step3()
        
        # Act & Assert - Should not raise
        verify_call_order(mock, ['step1', 'step2', 'step3'])
    
    def test_should_allow_extra_calls_in_sequence(self):
        """Should allow extra calls between expected calls"""
        # Arrange
        mock = create_mock()
        mock.step1()
        mock.unrelated_call()
        mock.step2()
        mock.another_unrelated_call()
        mock.step3()
        
        # Act & Assert - Should not raise
        verify_call_order(mock, ['step1', 'step2', 'step3'])
    
    def test_should_fail_for_wrong_order(self):
        """Should fail when methods called in wrong order"""
        # Arrange
        mock = create_mock()
        mock.step2()
        mock.step1()  # Wrong order
        mock.step3()
        
        # Act & Assert
        with pytest.raises(AssertionError, match="Expected call order not satisfied"):
            verify_call_order(mock, ['step1', 'step2', 'step3'])
    
    def test_should_fail_for_missing_calls(self):
        """Should fail when expected calls are missing"""
        # Arrange
        mock = create_mock()
        mock.step1()
        # Missing step2
        mock.step3()
        
        # Act & Assert
        with pytest.raises(AssertionError, match="Expected call order not satisfied"):
            verify_call_order(mock, ['step1', 'step2', 'step3'])


class TestPartialMocks:
    """Test partial mocking functionality"""
    
    def setup_method(self):
        # Create a real object to partially mock
        class RealService:
            def real_method(self):
                return "real_implementation"
            
            def another_real_method(self):
                return "another_real_implementation"
        
        self.real_service = RealService()
    
    def test_should_create_partial_mock(self):
        """Should create partial mock that wraps real object"""
        # Act
        partial = create_partial_mock(self.real_service)
        
        # Assert
        assert partial._real_object is self.real_service
        assert isinstance(partial._mocked_methods, dict)
    
    def test_should_mock_specific_methods(self):
        """Should allow mocking specific methods while keeping others real"""
        # Arrange
        partial = create_partial_mock(self.real_service)
        
        # Act - Mock one method
        mock_method = partial.mock_method('real_method')
        mock_method.returns('mocked_result')
        
        # Assert
        assert self.real_service.real_method() == 'mocked_result'
        assert self.real_service.another_real_method() == 'another_real_implementation'
        assert mock_method.called
    
    def test_should_restore_original_methods(self):
        """Should restore original methods when requested"""
        # Arrange
        partial = create_partial_mock(self.real_service)
        mock_method = partial.mock_method('real_method')
        mock_method.returns('mocked_result')
        
        # Verify mocked
        assert self.real_service.real_method() == 'mocked_result'
        
        # Act - Restore
        partial.restore_method('real_method')
        
        # Assert - Original behavior restored
        assert self.real_service.real_method() == 'real_implementation'
    
    def test_should_restore_all_methods(self):
        """Should restore all mocked methods at once"""
        # Arrange
        partial = create_partial_mock(self.real_service)
        partial.mock_method('real_method').returns('mocked1')
        partial.mock_method('another_real_method').returns('mocked2')
        
        # Act
        partial.restore_all()
        
        # Assert
        assert self.real_service.real_method() == 'real_implementation'
        assert self.real_service.another_real_method() == 'another_real_implementation'


class TestSpyFunctionality:
    """Test spy functionality for real objects"""
    
    def setup_method(self):
        class RealObject:
            def method_to_spy_on(self, arg):
                return f"real_result_{arg}"
        
        self.real_object = RealObject()
    
    def test_should_create_spy_on_real_method(self):
        """Should create spy that tracks calls to real method"""
        # Act
        spy = spy_on(self.real_object, 'method_to_spy_on')
        result = self.real_object.method_to_spy_on('test')
        
        # Assert - Real behavior preserved
        assert result == "real_result_test"
        
        # Spy recorded the call
        assert spy.called
        assert spy.call_count == 1
        assert spy.calls[0].args == ('test',)
        assert spy.calls[0].return_value == "real_result_test"
    
    def test_should_track_spy_exceptions(self):
        """Should track exceptions in spied methods"""
        # Arrange
        def failing_method():
            raise ValueError("Method failed")
        
        self.real_object.failing_method = failing_method
        spy = spy_on(self.real_object, 'failing_method')
        
        # Act & Assert
        with pytest.raises(ValueError, match="Method failed"):
            self.real_object.failing_method()
        
        # Spy should have recorded the exception
        assert spy.called
        assert isinstance(spy.calls[0].exception, ValueError)
    
    def test_should_restore_spied_method(self):
        """Should restore original method when spy is removed"""
        # Arrange
        original_result = self.real_object.method_to_spy_on('test')
        spy = spy_on(self.real_object, 'method_to_spy_on')
        
        # Verify spy is working
        spied_result = self.real_object.method_to_spy_on('test')
        assert spy.called
        assert spied_result == original_result  # Same behavior
        
        # Act
        restore_spy(spy)
        
        # Assert - Original behavior restored
        restored_result = self.real_object.method_to_spy_on('test') 
        assert restored_result == original_result
        # Spy should no longer track calls
        assert len(spy.calls) == 1  # Only the one call made while spying


class TestMockContext:
    """Test mock context manager for automatic cleanup"""
    
    def test_should_create_mocks_within_context(self):
        """Should create and manage mocks within context"""
        with mock_context() as ctx:
            mock1 = ctx.mock(name="TestMock1")
            mock2 = ctx.mock(name="TestMock2")
            
            # Use mocks
            mock1.test_method()
            mock2.test_method()
            
            assert mock1.test_method.called
            assert mock2.test_method.called
        
        # After context, mocks should be reset
        assert not mock1.test_method.called
        assert not mock2.test_method.called
    
    def test_should_create_spies_within_context(self):
        """Should create and restore spies within context"""
        # Arrange
        class TestObject:
            def method(self):
                return "original"
        
        obj = TestObject()
        
        with mock_context() as ctx:
            spy = ctx.spy(obj, 'method')
            result = obj.method()
            assert spy.called
            assert result == "original"
        
        # After context, spy should be restored and behavior preserved
        restored_result = obj.method()
        assert restored_result == "original"
    
    def test_should_handle_partial_mocks_within_context(self):
        """Should create and restore partial mocks within context"""
        # Arrange
        class TestObject:
            def method(self):
                return "original"
        
        obj = TestObject()
        
        with mock_context() as ctx:
            partial = ctx.partial_mock(obj)
            partial.mock_method('method').returns('mocked')
            
            assert obj.method() == 'mocked'
        
        # After context, should be restored
        assert obj.method() == 'original'


if __name__ == '__main__':
    # Run tests directly
    def run_all_tests():
        test_classes = [
            TestMockBehavior,
            TestCallOrderVerification,
            TestPartialMocks,
            TestSpyFunctionality,
            TestMockContext
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            instance = test_class()
            
            # Get all test methods
            test_methods = [method for method in dir(instance) 
                          if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                method = getattr(instance, method_name)
                
                try:
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    method()
                    print(f"✓ {test_class.__name__}.{method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"✗ {test_class.__name__}.{method_name}: {e}")
        
        print(f"\nMock Framework Tests: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests
    
    # Run the tests
    success = run_all_tests()
    if success:
        print("All mock framework tests passed!")
    else:
        print("Some mock framework tests failed!")