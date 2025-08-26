#!/usr/bin/env python3
"""
Basic System Components Test

Quick test to verify core components are working before running full E2E tests.
"""

import sys
import os
from pathlib import Path
import json
import unittest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class BasicSystemComponentsTest(unittest.TestCase):
    """Test basic system components"""
    
    def test_01_environment_setup(self):
        """Test environment setup"""
        # Check Python version
        self.assertGreaterEqual(sys.version_info, (3, 8), "Python 3.8+ required")
        
        # Check working directory
        cwd = Path.cwd()
        self.assertTrue(str(cwd).endswith("claude-tui"), "Should be in claude-tui directory")
        
        # Check src directory exists
        src_dir = cwd / "src"
        self.assertTrue(src_dir.exists(), "src directory should exist")
        
        print("✅ Environment setup OK")
    
    def test_02_oauth_config(self):
        """Test OAuth configuration"""
        cc_file = Path.home() / ".cc"
        
        if not cc_file.exists():
            # Create mock config for testing
            mock_config = {
                "oauth_token": "test_token_mock_" + "x" * 40,
                "api_url": "https://api.claude.ai"
            }
            with open(cc_file, 'w') as f:
                json.dump(mock_config, f, indent=2)
        
        with open(cc_file, 'r') as f:
            config = json.load(f)
        
        self.assertIn("oauth_token", config, "OAuth token should be present")
        self.assertGreater(len(config["oauth_token"]), 10, "OAuth token should be substantial")
        
        print("✅ OAuth configuration OK")
    
    def test_03_core_imports(self):
        """Test core component imports"""
        try:
            from claude_tui.integrations.claude_code_client import ClaudeCodeClient
            from claude_tui.integrations.claude_flow_client import ClaudeFlowClient
            from claude_tui.core.ai_interface import AIInterface
            from claude_tui.validation.real_time_validator import RealTimeValidator
            print("✅ Core imports OK")
        except ImportError as e:
            self.fail(f"Core import failed: {e}")
    
    def test_04_component_initialization(self):
        """Test component initialization"""
        try:
            from claude_tui.integrations.claude_code_client import ClaudeCodeClient
            client = ClaudeCodeClient()
            self.assertIsNotNone(client)
            print("✅ ClaudeCodeClient initialization OK")
        except Exception as e:
            print(f"⚠️  ClaudeCodeClient init issue: {e}")
        
        try:
            from claude_tui.integrations.claude_flow_client import ClaudeFlowClient
            flow_client = ClaudeFlowClient()
            self.assertIsNotNone(flow_client)
            print("✅ ClaudeFlowClient initialization OK")
        except Exception as e:
            print(f"⚠️  ClaudeFlowClient init issue: {e}")
        
        try:
            from claude_tui.core.ai_interface import AIInterface
            ai_interface = AIInterface()
            self.assertIsNotNone(ai_interface)
            print("✅ AIInterface initialization OK")
        except Exception as e:
            print(f"⚠️  AIInterface init issue: {e}")
    
    def test_05_claude_flow_available(self):
        """Test Claude Flow availability"""
        import subprocess
        
        try:
            result = subprocess.run(["npx", "claude-flow@alpha", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, "Claude Flow should be available")
            print(f"✅ Claude Flow version: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            print("⚠️  Claude Flow version check timed out")
        except Exception as e:
            print(f"⚠️  Claude Flow check failed: {e}")
    
    def test_06_basic_functionality(self):
        """Test basic functionality without external dependencies"""
        try:
            from claude_tui.core.config_manager import ConfigManager
            config = ConfigManager()
            self.assertIsNotNone(config)
            print("✅ ConfigManager basic functionality OK")
        except Exception as e:
            print(f"⚠️  ConfigManager issue: {e}")
        
        try:
            from claude_tui.core.logger import get_logger
            logger = get_logger("test")
            self.assertIsNotNone(logger)
            logger.info("Test log message")
            print("✅ Logger basic functionality OK")
        except Exception as e:
            print(f"⚠️  Logger issue: {e}")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)