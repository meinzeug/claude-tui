#!/usr/bin/env python3
"""
System Validation Script for Claude TUI.

Validates that all major components are working correctly and can integrate together.
This script tests the complete implementation pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from claude_tui.core.config_manager import ConfigManager
from claude_tui.services.validation_service import ValidationService, ValidationLevel
from claude_tui.validation.placeholder_detector import PlaceholderDetector
from claude_tui.validation.semantic_analyzer import SemanticAnalyzer
from claude_tui.validation.auto_completion_engine import AutoCompletionEngine


async def test_component_initialization():
    """Test that all major components can be initialized."""
    print("🔧 Testing Component Initialization...")
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        await config_manager.initialize()
        print("✅ ConfigManager initialized")
        
        # Initialize ValidationService
        validation_service = ValidationService()
        await validation_service._initialize_impl()
        print("✅ ValidationService initialized")
        
        # Initialize PlaceholderDetector
        placeholder_detector = PlaceholderDetector(config_manager)
        await placeholder_detector.initialize()
        print("✅ PlaceholderDetector initialized")
        
        # Initialize SemanticAnalyzer
        semantic_analyzer = SemanticAnalyzer(config_manager)
        await semantic_analyzer.initialize()
        print("✅ SemanticAnalyzer initialized")
        
        # Initialize AutoCompletionEngine
        auto_completion = AutoCompletionEngine(config_manager)
        await auto_completion.initialize()
        print("✅ AutoCompletionEngine initialized")
        
        return {
            'config_manager': config_manager,
            'validation_service': validation_service,
            'placeholder_detector': placeholder_detector,
            'semantic_analyzer': semantic_analyzer,
            'auto_completion': auto_completion
        }
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return None


async def test_placeholder_detection(components):
    """Test placeholder detection functionality."""
    print("\n🔍 Testing Placeholder Detection...")
    
    test_code = '''
def incomplete_function():
    """Function with placeholders."""
    # TODO: Implement this function
    pass

class IncompleteClass:
    # FIXME: Add proper implementation
    def method(self):
        raise NotImplementedError("Not implemented yet")
        
def another_function():
    # PLACEHOLDER: Add logic here
    ...
    '''
    
    try:
        detector = components['placeholder_detector']
        issues = await detector.detect_placeholders(test_code)
        
        print(f"✅ Detected {len(issues)} placeholder issues:")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue.description} (severity: {issue.severity.value})")
        
        if len(issues) > 3:
            print(f"   ... and {len(issues) - 3} more issues")
        
        return len(issues) >= 3  # Should find multiple placeholders
        
    except Exception as e:
        print(f"❌ Placeholder detection failed: {e}")
        return False


async def test_semantic_analysis(components):
    """Test semantic analysis functionality."""
    print("\n🧠 Testing Semantic Analysis...")
    
    test_code = '''
import unused_module
import os
import sys

def problematic_function():
    password = "hardcoded_password"  # Security issue
    eval("dangerous_code()")         # Security issue
    
    unused_var = "never used"
    
    for i in range(len(sys.argv)):   # Performance issue
        print(sys.argv[i])
    
    return None
    print("unreachable code")        # Unreachable code
    '''
    
    try:
        analyzer = components['semantic_analyzer']
        issues = await analyzer.analyze_content(test_code, language="python")
        
        print(f"✅ Detected {len(issues)} semantic issues:")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue.description} (type: {issue.issue_type})")
        
        if len(issues) > 3:
            print(f"   ... and {len(issues) - 3} more issues")
        
        return len(issues) >= 2  # Should find multiple semantic issues
        
    except Exception as e:
        print(f"❌ Semantic analysis failed: {e}")
        return False


async def test_validation_service(components):
    """Test comprehensive validation service."""
    print("\n🛡️ Testing Validation Service...")
    
    test_code = '''
def mixed_issues_function(param):
    """Function with multiple types of issues."""
    # TODO: Implement parameter validation
    
    if param is None:
        raise NotImplementedError("Validation not implemented")
    
    # Security issues
    secret_key = "hardcoded_secret_123"
    result = eval(f"process({param})")
    
    # Performance issue
    data = []
    for i in range(len(param)):
        data.append(param[i])
    
    # Unused import would be detected if we had one
    unused_var = "never used"
    
    return result  # May be undefined due to eval
    '''
    
    try:
        service = components['validation_service']
        result = await service.validate_code(
            code=test_code,
            language="python",
            validation_level=ValidationLevel.COMPREHENSIVE,
            check_placeholders=True,
            check_syntax=True,
            check_quality=True
        )
        
        print(f"✅ Validation completed:")
        print(f"   - Valid: {result['is_valid']}")
        print(f"   - Score: {result['score']:.2f}")
        print(f"   - Issues: {len(result['issues'])}")
        print(f"   - Warnings: {len(result['warnings'])}")
        print(f"   - Categories: {list(result['categories'].keys())}")
        
        # Should detect issues in multiple categories
        expected_categories = ['placeholder', 'security', 'quality']
        found_categories = list(result['categories'].keys())
        
        return any(cat in found_categories for cat in expected_categories)
        
    except Exception as e:
        print(f"❌ Validation service failed: {e}")
        return False


async def test_auto_completion(components):
    """Test auto-completion engine."""
    print("\n🔧 Testing Auto-Completion Engine...")
    
    try:
        engine = components['auto_completion']
        
        # Test code completion
        partial_code = "def incomplete_function():"
        suggestions = await engine.suggest_completion(
            partial_code, 
            language="python"
        )
        
        print(f"✅ Generated {len(suggestions)} code completion suggestions")
        
        if suggestions:
            print(f"   - Example suggestion: {suggestions[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-completion failed: {e}")
        return False


async def test_integration_workflow(components):
    """Test complete integration workflow."""
    print("\n🔄 Testing Integration Workflow...")
    
    test_code = '''
def example_workflow():
    """Example function with various issues for testing."""
    # TODO: Add input validation
    
    def process_data(data):
        # FIXME: Implement proper data processing
        if data is None:
            raise NotImplementedError()
        
        # Security issue
        password = "admin123"
        result = eval(data)  # Dangerous
        
        # Performance issue  
        processed = []
        for i in range(len(result)):
            processed.append(result[i])
        
        return processed
    
    # Call the function
    return process_data("test_data")
    '''
    
    try:
        # Step 1: Detect placeholders
        placeholder_issues = await components['placeholder_detector'].detect_placeholders(test_code)
        
        # Step 2: Analyze semantics
        semantic_issues = await components['semantic_analyzer'].analyze_content(
            test_code, language="python"
        )
        
        # Step 3: Comprehensive validation
        validation_result = await components['validation_service'].validate_code(
            code=test_code,
            language="python",
            validation_level=ValidationLevel.COMPREHENSIVE
        )
        
        # Step 4: Get completion suggestions
        suggestions = await components['auto_completion'].suggest_completion(
            test_code, language="python"
        )
        
        print(f"✅ Integration workflow completed:")
        print(f"   - Placeholder issues: {len(placeholder_issues)}")
        print(f"   - Semantic issues: {len(semantic_issues)}")
        print(f"   - Overall validation score: {validation_result['score']:.2f}")
        print(f"   - Completion suggestions: {len(suggestions)}")
        
        # Should detect issues across multiple components
        total_issues = len(placeholder_issues) + len(semantic_issues)
        return total_issues >= 3 and validation_result['score'] < 1.0
        
    except Exception as e:
        print(f"❌ Integration workflow failed: {e}")
        return False


async def test_performance_benchmarks(components):
    """Test performance benchmarks."""
    print("\n⚡ Testing Performance Benchmarks...")
    
    import time
    
    # Large test code
    large_code = "\n".join([
        f"def function_{i}(param):",
        f'    """Function number {i}"""',
        f"    # TODO: Implement function_{i}",
        f"    result = param * {i}",
        f"    return result",
        ""
    ] for i in range(50))
    
    try:
        start_time = time.time()
        
        # Run validation on large code
        result = await components['validation_service'].validate_code(
            code=large_code,
            language="python",
            validation_level=ValidationLevel.STANDARD
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Performance benchmark completed:")
        print(f"   - Code lines: {len(large_code.splitlines())}")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Issues detected: {len(result['issues'])}")
        print(f"   - Performance: {len(large_code.splitlines()) / processing_time:.0f} lines/second")
        
        return processing_time < 30  # Should complete within 30 seconds
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


async def cleanup_components(components):
    """Clean up all components."""
    print("\n🧹 Cleaning up components...")
    
    try:
        for name, component in components.items():
            if hasattr(component, 'cleanup'):
                await component.cleanup()
        
        print("✅ All components cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


async def main():
    """Main validation function."""
    print("🚀 Claude TUI System Validation")
    print("=" * 50)
    
    # Test results
    test_results = []
    components = None
    
    try:
        # Initialize components
        components = await test_component_initialization()
        if components is None:
            print("\n❌ Cannot continue - component initialization failed")
            return False
        
        test_results.append(("Component Initialization", True))
        
        # Run individual component tests
        tests = [
            ("Placeholder Detection", test_placeholder_detection),
            ("Semantic Analysis", test_semantic_analysis), 
            ("Validation Service", test_validation_service),
            ("Auto Completion", test_auto_completion),
            ("Integration Workflow", test_integration_workflow),
            ("Performance Benchmarks", test_performance_benchmarks)
        ]
        
        for test_name, test_func in tests:
            try:
                result = await test_func(components)
                test_results.append((test_name, result))
            except Exception as e:
                print(f"❌ Test {test_name} encountered error: {e}")
                test_results.append((test_name, False))
        
    finally:
        # Cleanup
        if components:
            await cleanup_components(components)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / len(test_results) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System validation successful!")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. System needs attention.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)