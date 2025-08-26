

# Development utilities
def _create_sample_tests():
    """Create sample tests for development and verification"""
    
    def test_framework_basic_functionality():
        """Test basic framework functionality"""
        # Test assertions
        expect(42).to.equal(42)
        expect([1, 2, 3]).to.have.length(3)
        expect({'key': 'value'}).to.have.property('key')
        
        # Test mock creation
        mock_obj = create_mock()
        mock_obj.some_method('arg1')
        expect(mock_obj.some_method).to.have.property('called').equal(True)
        
        return True
    
    async def test_framework_async_functionality():
        """Test async framework functionality"""
        async def async_operation():
            await asyncio.sleep(0.001)
            return "async result"
        
        # Test async assertions
        result = await expect(async_operation()).to.eventually.equal("async result")
        return True
    
    def test_framework_error_handling():
        """Test framework error handling"""
        def failing_function():
            raise ValueError("Test error")
        
        # Test exception assertions
        expect(failing_function).to.throws(ValueError, "Test error")
        return True
    
    return [
        test_framework_basic_functionality,
        test_framework_async_functionality, 
        test_framework_error_handling
    ]


if __name__ == "__main__":
    # Run sample tests when module is executed directly
    import asyncio
    
    print("Running Test Workflow Framework verification...")
    
    sample_tests = _create_sample_tests()
    success = run_quick_tests(*sample_tests)
    
    if success:
        print("\n🎉 Test Workflow Framework is working correctly!")
    else:
        print("\n❌ Test Workflow Framework has issues.")
        
    # Also test discovery
    print("\nTesting test discovery...")
    try:
        discovery = TestDiscovery()
        test_files = discovery.find_test_files()
        print(f"Found {len(test_files)} test files")
        
        if test_files:
            tests = discovery.load_tests(test_files[:1])  # Load from first file only
            print(f"Loaded {len(tests)} tests from first file")
        
        print("✓ Test discovery working")
    except Exception as e:
        print(f"✗ Test discovery failed: {e}")
