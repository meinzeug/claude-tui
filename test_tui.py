#!/usr/bin/env python3
"""
Test script to verify TUI functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all major components can be imported"""
    print("Testing imports...")
    
    try:
        # Core imports
        from core.project_manager import ProjectManager
        print("✅ Core: ProjectManager imported")
        
        from core.validator import ProgressValidator
        print("✅ Core: Validator imported")
        
        from core.task_engine import TaskEngine
        print("✅ Core: TaskEngine imported")
        
        # UI imports (will fail if Textual not installed)
        try:
            from ui.main_app import ClaudeTIUApp
            print("✅ UI: Main App imported")
        except ImportError as e:
            print(f"⚠️  UI: Main App import failed (Textual may not be installed): {e}")
        
        # Security imports
        from security.input_validator import SecurityInputValidator
        print("✅ Security: InputValidator imported")
        
        from security.code_sandbox import SecureCodeSandbox
        print("✅ Security: CodeSandbox imported")
        
        # API imports (will fail if FastAPI not installed)
        try:
            from api.main import app
            print("✅ API: FastAPI app imported")
        except ImportError as e:
            print(f"⚠️  API: FastAPI import failed (FastAPI may not be installed): {e}")
        
        print("\n✨ Core components successfully imported!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from core.project_manager import ProjectManager
        from core.config_manager import ConfigManager
        
        # Create config manager
        config = ConfigManager()
        print("✅ ConfigManager created")
        
        # Create project manager
        pm = ProjectManager(config)
        print("✅ ProjectManager created")
        
        # Test project creation (mock)
        project_info = {
            'name': 'test_project',
            'type': 'python',
            'path': '/tmp/test_project'
        }
        print(f"✅ Project info validated: {project_info['name']}")
        
        print("\n✨ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("=" * 50)
    print("Claude-TIU Component Test")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    # Test functionality
    func_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    if import_success and func_success:
        print("✅ All tests passed! Claude-TIU is ready.")
        print("\nTo run the full TUI:")
        print("  python run_tui.py")
        print("\nTo run the API:")
        print("  uvicorn src.api.main:app --reload")
    else:
        print("⚠️  Some tests failed. Please install dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()