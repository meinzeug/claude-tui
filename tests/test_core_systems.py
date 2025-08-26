#!/usr/bin/env python3
"""
Test Core Systems - Validate that all core systems can initialize properly
"""

import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_core_config_manager():
    """Test the core configuration manager"""
    try:
        print("=== Testing Core Config Manager ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from claude_tui.core.config_manager import ConfigManager
        
        config = ConfigManager()
        logger.info("✓ ConfigManager imported and instantiated")
        
        # Test basic operations
        config.config_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Config directory: {config.config_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Core config manager test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_core_project_manager():
    """Test the core project manager"""
    try:
        print("\n=== Testing Core Project Manager ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from claude_tui.core.config_manager import ConfigManager
        from claude_tui.core.project_manager import ProjectManager
        
        config = ConfigManager()
        project_manager = ProjectManager(config)
        
        logger.info("✓ ProjectManager imported and instantiated")
        
        # Test basic operations
        if hasattr(project_manager, 'current_project'):
            logger.info(f"✓ Current project: {project_manager.current_project}")
        
        return True
        
    except Exception as e:
        print(f"✗ Core project manager test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_ai_interface():
    """Test the AI interface"""
    try:
        print("\n=== Testing AI Interface ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from claude_tui.core.config_manager import ConfigManager
        from claude_tui.integrations.ai_interface import AIInterface
        
        config = ConfigManager()
        ai_interface = AIInterface(config)
        
        logger.info("✓ AIInterface imported and instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ AI interface test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_validation_engine():
    """Test the validation engine"""
    try:
        print("\n=== Testing Validation Engine ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        from claude_tui.core.config_manager import ConfigManager
        from claude_tui.core.progress_validator import ProgressValidator
        
        config = ConfigManager()
        validator = ProgressValidator(config)
        
        logger.info("✓ ProgressValidator imported and instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation engine test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_integration_components():
    """Test integration components"""
    try:
        print("\n=== Testing Integration Components ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        # Test key integration components
        from claude_tui.integrations.claude_code_client import ClaudeCodeClient
        from claude_tui.integrations.claude_flow_client import ClaudeFlowClient
        from claude_tui.integrations.anti_hallucination_integration import AntiHallucinationIntegration
        
        logger.info("✓ Integration components imported successfully")
        
        # Test instantiation with minimal config
        config = {"base_url": "https://api.claude.ai/v1"}
        
        client = ClaudeCodeClient(config)
        logger.info("✓ ClaudeCodeClient instantiated")
        
        flow_client = ClaudeFlowClient()  
        logger.info("✓ ClaudeFlowClient instantiated")
        
        anti_hallucination = AntiHallucinationIntegration()
        logger.info("✓ AntiHallucinationIntegration instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration components test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_validation_components():
    """Test validation components"""
    try:
        print("\n=== Testing Validation Components ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        # Test validation components
        from claude_tui.validation.anti_hallucination_engine import AntiHallucinationEngine
        from claude_tui.validation.real_time_validator import RealTimeValidator
        from claude_tui.validation.placeholder_detector import PlaceholderDetector
        
        logger.info("✓ Validation components imported successfully")
        
        # Test instantiation
        engine = AntiHallucinationEngine()
        logger.info("✓ AntiHallucinationEngine instantiated")
        
        validator = RealTimeValidator()
        logger.info("✓ RealTimeValidator instantiated")
        
        detector = PlaceholderDetector()
        logger.info("✓ PlaceholderDetector instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation components test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_performance_components():
    """Test performance monitoring components"""
    try:
        print("\n=== Testing Performance Components ===")
        sys.path.insert(0, '/home/tekkadmin/claude-tui/src')
        
        # Test performance components
        from performance.memory_optimizer import MemoryOptimizer
        from performance.performance_test_suite import PerformanceTestSuite
        
        logger.info("✓ Performance components imported successfully")
        
        # Test instantiation
        optimizer = MemoryOptimizer()
        logger.info("✓ MemoryOptimizer instantiated")
        
        test_suite = PerformanceTestSuite()
        logger.info("✓ PerformanceTestSuite instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance components test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all core system tests"""
    print("🧪 Starting Core Systems Tests")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Core Config Manager", test_core_config_manager),
        ("Core Project Manager", test_core_project_manager),
        ("AI Interface", test_ai_interface),
        ("Validation Engine", test_validation_engine),
        ("Integration Components", test_integration_components),
        ("Validation Components", test_validation_components),
        ("Performance Components", test_performance_components),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CORE SYSTEMS TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    return {
        'tests_passed': passed,
        'tests_total': len(tests),
        'success_rate': passed / len(tests),
        'results': results
    }

if __name__ == '__main__':
    results = main()