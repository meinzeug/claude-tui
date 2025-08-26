"""Unit Tests for Assertions - London School TDD

Tests the chainable assertion API and async assertion capabilities.
"""

import asyncio
import pytest
from src.assertions import (
    expect, AssertionError, assert_equal, assert_deep_equal,
    assert_throws, assert_eventually_equal
)


class TestBasicAssertions:
    """Test basic assertion functionality"""
    
    def test_should_assert_equality(self):
        """Should assert equality with fluent interface"""
        # Should pass
        expect(42).to.equal(42)
        expect("test").to.equal("test")
        expect([1, 2, 3]).to.equal([1, 2, 3])
    
    def test_should_fail_on_inequality(self):
        """Should fail when values are not equal"""
        with pytest.raises(AssertionError, match="Expected 42 to equal 43"):
            expect(42).to.equal(43)
    
    def test_should_support_negation(self):
        """Should support negated assertions"""
        expect(42).not_.to.equal(43)
        expect("hello").not_.to.equal("world")
    
    def test_should_fail_negated_equality(self):
        """Should fail when negated equality is wrong"""
        with pytest.raises(AssertionError, match="Expected 42 not to equal 42"):
            expect(42).not_.to.equal(42)


class TestChainableInterface:
    """Test the chainable fluent interface"""
    
    def test_should_chain_multiple_assertions(self):
        """Should support chaining multiple assertions"""
        expect(42).to.equal(42).and_be.a('int').and_be.greater_than(40)
        expect([1, 2, 3]).to.have.length(3).and_contain(2)
    
    def test_should_support_grammatical_sugar(self):
        """Should support grammatical sugar words"""
        result = expect(42).to.be.a('int')
        assert result is not None  # Should return chainable object
        
        result = expect(42).have.been.equal(42)
        assert result is not None


class TestDeepEquality:
    """Test deep equality assertions"""
    
    def test_should_assert_deep_equality(self):
        """Should assert deep equality for complex objects"""
        obj1 = {'nested': {'value': [1, 2, {'deep': True}]}}
        obj2 = {'nested': {'value': [1, 2, {'deep': True}]}}
        
        expect(obj1).to.deep_equal(obj2)
    
    def test_should_fail_deep_inequality(self):
        """Should fail when objects are not deeply equal"""
        obj1 = {'nested': {'value': [1, 2, {'deep': True}]}}
        obj2 = {'nested': {'value': [1, 2, {'deep': False}]}}
        
        with pytest.raises(AssertionError, match="to deep equal"):
            expect(obj1).to.deep_equal(obj2)
    
    def test_should_handle_different_types(self):
        """Should fail when types are different"""
        with pytest.raises(AssertionError):
            expect([1, 2, 3]).to.deep_equal((1, 2, 3))


class TestTypeAssertions:
    """Test type checking assertions"""
    
    def test_should_assert_type_by_class(self):
        """Should assert type using class"""
        expect(42).to.be.a(int)
        expect("test").to.be.a(str)
        expect([1, 2, 3]).to.be.a(list)
    
    def test_should_assert_type_by_name(self):
        """Should assert type using type name string"""
        expect(42).to.be.a('int')
        expect("test").to.be.a('str')
        expect([1, 2, 3]).to.be.a('list')
    
    def test_should_fail_wrong_type(self):
        """Should fail when type is wrong"""
        with pytest.raises(AssertionError, match="to be a str, but got int"):
            expect(42).to.be.a('str')


class TestNumericalAssertions:
    """Test numerical comparison assertions"""
    
    def test_should_assert_greater_than(self):
        """Should assert greater than"""
        expect(42).to.be.greater_than(40)
        expect(3.14).to.be.greater_than(3)
    
    def test_should_fail_not_greater_than(self):
        """Should fail when not greater than"""
        with pytest.raises(AssertionError, match="to be greater than"):
            expect(40).to.be.greater_than(42)
    
    def test_should_assert_less_than(self):
        """Should assert less than"""
        expect(40).to.be.less_than(42)
        expect(3.0).to.be.less_than(3.14)
    
    def test_should_fail_not_less_than(self):
        """Should fail when not less than"""
        with pytest.raises(AssertionError, match="to be less than"):
            expect(42).to.be.less_than(40)


class TestContainerAssertions:
    """Test container-related assertions"""
    
    def test_should_assert_contains(self):
        """Should assert container contains item"""
        expect([1, 2, 3]).to.contain(2)
        expect("hello world").to.contain("world")
        expect({'a': 1, 'b': 2}).to.contain('a')
    
    def test_should_fail_not_contains(self):
        """Should fail when container doesn't contain item"""
        with pytest.raises(AssertionError, match="to contain"):
            expect([1, 2, 3]).to.contain(4)
    
    def test_should_assert_length(self):
        """Should assert container length"""
        expect([1, 2, 3]).to.have.length(3)
        expect("hello").to.have.length(5)
        expect({'a': 1, 'b': 2}).to.have.length(2)
    
    def test_should_fail_wrong_length(self):
        """Should fail when length is wrong"""
        with pytest.raises(AssertionError, match="to have length 5, but got 3"):
            expect([1, 2, 3]).to.have.length(5)


class TestPropertyAssertions:
    """Test property-related assertions"""
    
    def test_should_assert_has_property(self):
        """Should assert object has property"""
        obj = {'key': 'value', 'number': 42}
        expect(obj).to.have.property('key')
        expect(obj).to.have.property('number')
    
    def test_should_fail_missing_property(self):
        """Should fail when property is missing"""
        obj = {'key': 'value'}
        with pytest.raises(AssertionError, match="to have property 'missing'"):
            expect(obj).to.have.property('missing')
    
    def test_should_assert_property_value(self):
        """Should assert property has specific value"""
        obj = {'key': 'value', 'number': 42}
        expect(obj).to.have.property('key').with_value('value')
        expect(obj).to.have.property('number').with_value(42)
    
    def test_should_assert_property_type(self):
        """Should assert property type"""
        obj = {'key': 'value', 'number': 42}
        expect(obj).to.have.property('key').of_type(str)
        expect(obj).to.have.property('number').of_type('int')


class TestExceptionAssertions:
    """Test exception-related assertions"""
    
    def test_should_assert_throws_exception(self):
        """Should assert function throws exception"""
        def failing_function():
            raise ValueError("Test error")
        
        expect(failing_function).to.throws()
    
    def test_should_assert_specific_exception_type(self):
        """Should assert specific exception type"""
        def failing_function():
            raise ValueError("Test error")
        
        expect(failing_function).to.throws(ValueError)
    
    def test_should_assert_exception_message(self):
        """Should assert exception message pattern"""
        def failing_function():
            raise ValueError("Test error message")
        
        expect(failing_function).to.throws(ValueError, "Test error")
    
    def test_should_fail_when_no_exception(self):
        """Should fail when function doesn't throw"""
        def passing_function():
            return "success"
        
        with pytest.raises(AssertionError, match="to throw an exception"):
            expect(passing_function).to.throws()
    
    def test_should_fail_wrong_exception_type(self):
        """Should fail when wrong exception type is thrown"""
        def failing_function():
            raise TypeError("Wrong type")
        
        with pytest.raises(AssertionError, match="to throw ValueError"):
            expect(failing_function).to.throws(ValueError)


class TestAsyncAssertions:
    """Test async assertion capabilities"""
    
    async def test_should_assert_async_equality(self):
        """Should assert equality for async operations"""
        async def async_operation():
            await asyncio.sleep(0.001)
            return "async result"
        
        await expect(async_operation()).to.eventually.equal("async result")
    
    async def test_should_fail_async_inequality(self):
        """Should fail async equality assertions"""
        async def async_operation():
            await asyncio.sleep(0.001)
            return "async result"
        
        with pytest.raises(AssertionError):
            await expect(async_operation()).to.eventually.equal("wrong result")
    
    async def test_should_assert_async_rejection(self):
        """Should assert async operations reject"""
        async def failing_async():
            await asyncio.sleep(0.001)
            raise ValueError("Async error")
        
        await expect(failing_async()).to.eventually.reject(ValueError)
    
    async def test_should_fail_async_no_rejection(self):
        """Should fail when async operation doesn't reject"""
        async def passing_async():
            await asyncio.sleep(0.001)
            return "success"
        
        with pytest.raises(AssertionError, match="Expected operation to reject"):
            await expect(passing_async()).to.eventually.reject()


class TestConvenienceFunctions:
    """Test convenience assertion functions"""
    
    def test_assert_equal_function(self):
        """Should provide simple assert_equal function"""
        assert_equal(42, 42)
        assert_equal("test", "test")
    
    def test_assert_equal_with_message(self):
        """Should support custom error messages"""
        with pytest.raises(AssertionError, match="Custom message:"):
            assert_equal(42, 43, "Custom message")
    
    def test_assert_deep_equal_function(self):
        """Should provide simple deep equality function"""
        obj1 = {'nested': [1, 2, 3]}
        obj2 = {'nested': [1, 2, 3]}
        assert_deep_equal(obj1, obj2)
    
    def test_assert_throws_function(self):
        """Should provide simple throws assertion function"""
        def failing_fn():
            raise ValueError("Test")
        
        assert_throws(failing_fn, ValueError)
    
    async def test_assert_eventually_equal_function(self):
        """Should provide simple async equality function"""
        async def async_fn():
            await asyncio.sleep(0.001)
            return "result"
        
        await assert_eventually_equal(async_fn(), "result")


class TestErrorMessages:
    """Test error message clarity and detail"""
    
    def test_should_provide_detailed_error_messages(self):
        """Should provide clear, detailed error messages"""
        try:
            expect(42).to.equal(43)
        except AssertionError as e:
            assert "Expected 42 to equal 43" in str(e)
            assert "actual: 42" in str(e)
            assert "expected: 43" in str(e)
    
    def test_should_include_context_in_errors(self):
        """Should include context information in error messages"""
        try:
            expect([1, 2, 3]).to.have.length(5)
        except AssertionError as e:
            assert "actual: 3" in str(e)
            assert "expected: 5" in str(e)


if __name__ == '__main__':
    # Run tests directly
    import asyncio
    
    async def run_all_tests():
        test_classes = [
            TestBasicAssertions,
            TestChainableInterface,
            TestDeepEquality,
            TestTypeAssertions,
            TestNumericalAssertions,
            TestContainerAssertions,
            TestPropertyAssertions,
            TestExceptionAssertions,
            TestAsyncAssertions,
            TestConvenienceFunctions,
            TestErrorMessages
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
                    
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                    
                    print(f"✓ {test_class.__name__}.{method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"✗ {test_class.__name__}.{method_name}: {e}")
        
        print(f"\nAssertion Tests: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests
    
    # Run the tests
    success = asyncio.run(run_all_tests())
    if success:
        print("All assertion tests passed!")
    else:
        print("Some assertion tests failed!")