#!/usr/bin/env python3
"""
Basic Usage Example - Test Workflow Framework
Demonstrates basic usage patterns and integration of all components
"""

import asyncio
import sys
import os
from pathlib import Path

# Add test-workflow to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_workflow.core import (
    create_test_framework,
    TestSuite,
    TestContext,
    test_case,
    suite_setup,
    suite_teardown
)


# Example application code to test
class Calculator:
    """Simple calculator for demonstration"""
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


class UserService:
    """User service for demonstration"""
    
    def __init__(self, database):
        self.database = database
    
    def create_user(self, name, email):
        if not name or not email:
            raise ValueError("Name and email are required")
        
        user_id = len(self.database.get("users", [])) + 1
        user = {
            "id": user_id,
            "name": name,
            "email": email
        }
        
        if "users" not in self.database:
            self.database["users"] = []
        
        self.database["users"].append(user)
        return user
    
    def get_user(self, user_id):
        users = self.database.get("users", [])
        for user in users:
            if user["id"] == user_id:
                return user
        return None
    
    def list_users(self):
        return self.database.get("users", [])


async def run_basic_examples():
    """Run basic usage examples"""
    
    print("🚀 Running Test Workflow Framework - Basic Usage Examples")
    print("=" * 60)
    
    # Create test framework with multiple reporters
    framework = create_test_framework(
        console_output=True,
        json_output="examples/results/basic_results.json",
        html_output="examples/results/basic_report.html"
    )
    
    # Example 1: Simple unit tests
    calculator_suite = create_calculator_tests()
    framework.add_suite(calculator_suite)
    
    # Example 2: Tests with mocks and context
    user_service_suite = create_user_service_tests()
    framework.add_suite(user_service_suite)
    
    # Example 3: Async tests
    async_suite = create_async_tests()
    framework.add_suite(async_suite)
    
    # Example 4: Error handling tests
    error_suite = create_error_handling_tests()
    framework.add_suite(error_suite)
    
    # Run all tests
    print("\n📋 Running all test suites...")
    results = await framework.run_all()
    
    print(f"\n📊 Summary:")
    print(f"  Total Tests: {results['summary']['total']}")
    print(f"  Passed: {results['summary']['passed']} ✅")
    print(f"  Failed: {results['summary']['failed']} ❌")
    print(f"  Errors: {results['summary']['errors']} ⚠️")
    print(f"  Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"  Duration: {results['summary']['duration']:.3f}s")
    
    return results


def create_calculator_tests():
    """Create test suite for Calculator class"""
    
    suite = TestSuite("Calculator Tests")
    
    def test_addition(assertions, mocks, context):
        """Test calculator addition"""
        calc = Calculator()
        
        # Test basic addition
        result = calc.add(2, 3)
        assertions.equal(result, 5, "2 + 3 should equal 5")
        
        # Test with negative numbers
        result = calc.add(-1, 1)
        assertions.equal(result, 0, "-1 + 1 should equal 0")
        
        # Test with floats
        result = calc.add(0.1, 0.2)
        assertions.that(result).is_greater_than(0.29)
        assertions.that(result).is_less_than(0.31)
        
    def test_subtraction(assertions, mocks, context):
        """Test calculator subtraction"""
        calc = Calculator()
        
        result = calc.subtract(5, 3)
        assertions.equal(result, 2)
        
        result = calc.subtract(0, 5)
        assertions.equal(result, -5)
        
    def test_multiplication(assertions, mocks, context):
        """Test calculator multiplication"""
        calc = Calculator()
        
        result = calc.multiply(4, 5)
        assertions.equal(result, 20)
        
        result = calc.multiply(-2, 3)
        assertions.equal(result, -6)
        
    def test_division(assertions, mocks, context):
        """Test calculator division"""
        calc = Calculator()
        
        result = calc.divide(10, 2)
        assertions.equal(result, 5.0)
        
        # Test division by zero
        with assertions.raises(ValueError, "Division by zero should raise ValueError"):
            calc.divide(10, 0)
            
    # Add tests to suite
    suite.add_test(test_addition, tags=["unit", "calculator"])
    suite.add_test(test_subtraction, tags=["unit", "calculator"])
    suite.add_test(test_multiplication, tags=["unit", "calculator"])
    suite.add_test(test_division, tags=["unit", "calculator"])
    
    return suite


def create_user_service_tests():
    """Create test suite for UserService class with mocks and context"""
    
    suite = TestSuite("User Service Tests")
    
    # Setup shared context
    def setup_suite(context):
        """Suite setup - runs once before all tests"""
        context.set("test_database", {})
        context.set("user_count", 0)
        
    def teardown_suite(context):
        """Suite teardown - runs once after all tests"""
        context.clear()
        
    def setup_each_test(context):
        """Before each test - runs before each test"""
        # Reset database for each test
        context.set("test_database", {"users": []})
        
    suite.setup = setup_suite
    suite.teardown = teardown_suite
    suite.before_each = setup_each_test
    
    def test_create_user(assertions, mocks, context):
        """Test user creation"""
        database = context.get("test_database")
        service = UserService(database)
        
        # Create a user
        user = service.create_user("John Doe", "john@example.com")
        
        # Assertions
        assertions.is_not_none(user)
        assertions.equal(user["name"], "John Doe")
        assertions.equal(user["email"], "john@example.com")
        assertions.equal(user["id"], 1)
        
        # Verify user was added to database
        users = service.list_users()
        assertions.equal(len(users), 1)
        assertions.equal(users[0], user)
        
    def test_create_user_validation(assertions, mocks, context):
        """Test user creation validation"""
        database = context.get("test_database")
        service = UserService(database)
        
        # Test missing name
        with assertions.raises(ValueError):
            service.create_user("", "john@example.com")
            
        # Test missing email
        with assertions.raises(ValueError):
            service.create_user("John Doe", "")
            
    def test_get_user(assertions, mocks, context):
        """Test user retrieval"""
        database = context.get("test_database")
        service = UserService(database)
        
        # Create a user first
        created_user = service.create_user("Jane Doe", "jane@example.com")
        
        # Retrieve the user
        retrieved_user = service.get_user(created_user["id"])
        
        assertions.equal(retrieved_user, created_user)
        
        # Test non-existent user
        non_existent = service.get_user(999)
        assertions.is_none(non_existent)
        
    def test_list_users(assertions, mocks, context):
        """Test listing users"""
        database = context.get("test_database")
        service = UserService(database)
        
        # Initially empty
        users = service.list_users()
        assertions.empty(users)
        
        # Add some users
        user1 = service.create_user("User 1", "user1@example.com")
        user2 = service.create_user("User 2", "user2@example.com")
        
        # List all users
        all_users = service.list_users()
        assertions.equal(len(all_users), 2)
        assertions.contains(all_users, user1)
        assertions.contains(all_users, user2)
        
    def test_user_service_with_mocks(assertions, mocks, context):
        """Test user service with database mocks"""
        
        # Create mock database
        mock_db = mocks.create_mock("mock_database")
        mock_db.get.return_value = []
        
        service = UserService(mock_db)
        
        # Mock database interactions
        users_list = service.list_users()
        
        # Verify mock was called
        mocks.assert_called_with("mock_database", "users", [])
        assertions.equal(users_list, [])
        
    # Add tests to suite
    suite.add_test(test_create_user, tags=["integration", "user_service"])
    suite.add_test(test_create_user_validation, tags=["unit", "validation"])
    suite.add_test(test_get_user, tags=["integration", "user_service"])
    suite.add_test(test_list_users, tags=["integration", "user_service"])
    suite.add_test(test_user_service_with_mocks, tags=["unit", "mocks"])
    
    return suite


def create_async_tests():
    """Create test suite for async functionality"""
    
    suite = TestSuite("Async Tests")
    
    async def test_basic_async_operation(assertions, mocks, context):
        """Test basic async operation"""
        
        async def async_function(value):
            await asyncio.sleep(0.01)  # Simulate async work
            return value * 2
            
        result = await async_function(5)
        assertions.equal(result, 10)
        
    async def test_async_with_mocks(assertions, mocks, context):
        """Test async operations with mocks"""
        
        # Create async mock
        async_service = mocks.create_async_mock(
            "async_service",
            return_value="async_result"
        )
        
        result = await async_service()
        assertions.equal(result, "async_result")
        
        # Verify mock was called
        mocks.assert_called("async_service")
        
    async def test_async_error_handling(assertions, mocks, context):
        """Test async error handling"""
        
        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async operation failed")
            
        # Test that async exception is properly caught
        with assertions.raises(RuntimeError):
            await failing_async_function()
            
    # Add async tests to suite
    suite.add_test(test_basic_async_operation, tags=["async", "unit"])
    suite.add_test(test_async_with_mocks, tags=["async", "mocks"])
    suite.add_test(test_async_error_handling, tags=["async", "error_handling"])
    
    return suite


def create_error_handling_tests():
    """Create test suite for error handling scenarios"""
    
    suite = TestSuite("Error Handling Tests")
    
    def test_expected_failure(assertions, mocks, context):
        """Test that demonstrates expected failure (for testing framework)"""
        # This test is meant to fail to show error handling
        assertions.equal(1, 2, "This test is expected to fail")
        
    def test_exception_handling(assertions, mocks, context):
        """Test exception handling"""
        
        def risky_function():
            raise ValueError("Something went wrong")
            
        # Test that exception is properly caught and reported
        with assertions.raises(ValueError) as exc_context:
            risky_function()
            
        # Can access the exception if needed
        assertions.that(str(exc_context.exception_caught)).contains("went wrong")
        
    def test_assertion_with_custom_message(assertions, mocks, context):
        """Test assertion with custom error message"""
        
        value = 42
        # This will pass
        assertions.equal(value, 42, f"Value should be 42, got {value}")
        
        # Test comprehensive assertions
        assertions.that(value).is_greater_than(40)
        assertions.that(value).is_less_than(50)
        assertions.isinstance_of(value, int)
        
    # Add error handling tests
    suite.add_test(test_expected_failure, tags=["error_demo"])
    suite.add_test(test_exception_handling, tags=["error_handling"])
    suite.add_test(test_assertion_with_custom_message, tags=["assertions"])
    
    return suite


if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("examples/results", exist_ok=True)
    
    # Run the examples
    results = asyncio.run(run_basic_examples())
    
    print(f"\n📄 Reports generated:")
    print(f"  JSON: examples/results/basic_results.json")
    print(f"  HTML: examples/results/basic_report.html")
    print(f"\n🎉 Example completed successfully!")
    
    # Return appropriate exit code
    if results['summary']['failed'] > 0 or results['summary']['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)