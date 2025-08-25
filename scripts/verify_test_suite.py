#!/usr/bin/env python3
"""
Test Suite Verification Script

Verifies that the comprehensive test suite is properly configured
and can be executed successfully.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util
import json
from typing import List, Dict, Any


class TestSuiteVerifier:
    """Verifies test suite configuration and dependencies."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.errors = []
        self.warnings = []
    
    def verify_all(self) -> bool:
        """Run all verification checks."""
        print("🔍 Verifying Claude TIU Test Suite Configuration")
        print("=" * 55)
        
        # Run all verification methods
        checks = [
            ("Test Directory Structure", self.verify_test_structure),
            ("Test Dependencies", self.verify_dependencies),
            ("Pytest Configuration", self.verify_pytest_config),
            ("Test File Syntax", self.verify_test_syntax),
            ("Test Fixtures", self.verify_test_fixtures),
            ("Mock Components", self.verify_mock_components),
            ("Coverage Configuration", self.verify_coverage_config)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                success = check_func()
                if success:
                    print(f"  ✅ {check_name}: PASSED")
                    passed_checks += 1
                else:
                    print(f"  ❌ {check_name}: FAILED")
            except Exception as e:
                print(f"  💥 {check_name}: ERROR - {e}")
                self.errors.append(f"{check_name}: {e}")
        
        # Print summary
        print("\n" + "=" * 55)
        print(f"📊 Verification Summary: {passed_checks}/{total_checks} checks passed")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        success = passed_checks == total_checks and len(self.errors) == 0
        
        if success:
            print("\n🎉 Test suite verification completed successfully!")
            print("   Ready to run: python scripts/test_runner.py")
        else:
            print("\n⚠️  Test suite has issues that need to be addressed.")
        
        return success
    
    def verify_test_structure(self) -> bool:
        """Verify test directory structure exists."""
        required_dirs = [
            "tests",
            "tests/unit",
            "tests/integration", 
            "tests/performance",
            "tests/validation",
            "tests/ui",
            "tests/fixtures"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.errors.append(f"Missing directories: {', '.join(missing_dirs)}")
            return False
        
        # Check for test files
        test_files = list(self.test_dir.rglob("test_*.py"))
        if len(test_files) < 5:  # Should have at least 5 test files
            self.warnings.append(f"Only found {len(test_files)} test files")
        
        print(f"    Found {len(test_files)} test files")
        return True
    
    def verify_dependencies(self) -> bool:
        """Verify required test dependencies are available."""
        required_packages = [
            "pytest",
            "pytest-asyncio", 
            "pytest-cov",
            "pytest-mock",
            "pytest-timeout",
            "pytest-benchmark",
            "pytest-html"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                # Try to import the package
                if package == "pytest-asyncio":
                    import pytest_asyncio
                elif package == "pytest-cov":
                    import pytest_cov
                elif package == "pytest-mock":
                    import pytest_mock
                elif package == "pytest-timeout":
                    import pytest_timeout
                elif package == "pytest-benchmark":
                    import pytest_benchmark
                elif package == "pytest-html":
                    import pytest_html
                else:
                    __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            self.errors.append("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print(f"    All {len(required_packages)} required packages available")
        return True
    
    def verify_pytest_config(self) -> bool:
        """Verify pytest configuration is valid."""
        pytest_ini = self.project_root / "pytest.ini"
        if not pytest_ini.exists():
            self.errors.append("pytest.ini configuration file missing")
            return False
        
        try:
            # Try to parse pytest configuration
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.errors.append(f"Pytest configuration error: {result.stderr}")
                return False
            
            print(f"    Pytest configuration is valid")
            return True
            
        except subprocess.TimeoutExpired:
            self.errors.append("Pytest configuration check timed out")
            return False
        except Exception as e:
            self.errors.append(f"Error checking pytest config: {e}")
            return False
    
    def verify_test_syntax(self) -> bool:
        """Verify all test files have valid Python syntax."""
        test_files = list(self.test_dir.rglob("test_*.py"))
        syntax_errors = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Try to compile the source
                compile(source, str(test_file), 'exec')
                
            except SyntaxError as e:
                syntax_errors.append(f"{test_file}: {e}")
            except Exception as e:
                self.warnings.append(f"Could not check {test_file}: {e}")
        
        if syntax_errors:
            self.errors.extend(syntax_errors)
            return False
        
        print(f"    All {len(test_files)} test files have valid syntax")
        return True
    
    def verify_test_fixtures(self) -> bool:
        """Verify test fixtures are properly configured."""
        fixtures_file = self.test_dir / "fixtures" / "comprehensive_test_fixtures.py"
        
        if not fixtures_file.exists():
            self.errors.append("Test fixtures file missing")
            return False
        
        try:
            # Try to import fixtures
            spec = importlib.util.spec_from_file_location("test_fixtures", fixtures_file)
            fixtures_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fixtures_module)
            
            # Check for required classes
            required_classes = ["TestDataGenerator", "MockComponents", "TestFileSystem"]
            missing_classes = []
            
            for class_name in required_classes:
                if not hasattr(fixtures_module, class_name):
                    missing_classes.append(class_name)
            
            if missing_classes:
                self.errors.append(f"Missing fixture classes: {', '.join(missing_classes)}")
                return False
            
            print(f"    Test fixtures properly configured with all required classes")
            return True
            
        except Exception as e:
            self.errors.append(f"Error loading test fixtures: {e}")
            return False
    
    def verify_mock_components(self) -> bool:
        """Verify mock components can be created."""
        try:
            # Import test utilities
            sys.path.insert(0, str(self.test_dir))
            from fixtures.comprehensive_test_fixtures import MockComponents
            
            # Try to create mock components
            mock_factory = MockComponents()
            
            # Test basic mock creation
            mock_project_manager = mock_factory.create_project_manager()
            mock_ai_interface = mock_factory.create_ai_interface()
            mock_validator = mock_factory.create_validator()
            
            if not all([mock_project_manager, mock_ai_interface, mock_validator]):
                self.errors.append("Failed to create required mock components")
                return False
            
            print(f"    Mock components can be created successfully")
            return True
            
        except Exception as e:
            self.errors.append(f"Error creating mock components: {e}")
            return False
        finally:
            # Clean up sys.path
            if str(self.test_dir) in sys.path:
                sys.path.remove(str(self.test_dir))
    
    def verify_coverage_config(self) -> bool:
        """Verify coverage configuration is valid."""
        try:
            # Check if coverage configuration is in pytest.ini
            pytest_ini = self.project_root / "pytest.ini"
            
            if not pytest_ini.exists():
                return False
            
            with open(pytest_ini, 'r') as f:
                config_content = f.read()
            
            # Check for coverage settings
            required_coverage_settings = [
                "--cov=",
                "--cov-report=",
                "--cov-fail-under="
            ]
            
            missing_settings = []
            for setting in required_coverage_settings:
                if setting not in config_content:
                    missing_settings.append(setting)
            
            if missing_settings:
                self.warnings.append(f"Missing coverage settings: {', '.join(missing_settings)}")
            
            print(f"    Coverage configuration found in pytest.ini")
            return True
            
        except Exception as e:
            self.warnings.append(f"Could not verify coverage config: {e}")
            return True  # Don't fail verification for this
    
    def quick_test_run(self) -> bool:
        """Run a quick test to verify the suite works."""
        print("\n🚀 Running quick test verification...")
        
        try:
            # Run a simple test collection
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--collect-only",
                "--quiet",
                "tests/"
            ], 
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30
            )
            
            if result.returncode == 0:
                print("  ✅ Test collection successful")
                return True
            else:
                print(f"  ❌ Test collection failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⚠️  Test collection timed out")
            return False
        except Exception as e:
            print(f"  💥 Error during test collection: {e}")
            return False


def main():
    """Main verification entry point."""
    verifier = TestSuiteVerifier()
    
    try:
        success = verifier.verify_all()
        
        # Run quick test if verification passed
        if success:
            print("\n" + "=" * 55)
            test_success = verifier.quick_test_run()
            if not test_success:
                success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Verification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()