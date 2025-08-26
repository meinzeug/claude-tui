"""
Developer Experience Module - Test Workflow Framework
Provides tools and utilities for smooth development experience
"""

import os
import sys
import importlib
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

from .exceptions import EnvironmentError, ConfigurationError


logger = logging.getLogger(__name__)


class DevelopmentEnvironment:
    """Manages development environment setup and validation"""
    
    def __init__(self):
        self.requirements = {
            "python": {
                "min_version": (3, 8),
                "current": sys.version_info[:2]
            },
            "packages": [
                "pytest",
                "pytest-asyncio", 
                "pytest-cov",
                "pytest-html"
            ],
            "optional_packages": [
                "black",
                "flake8",
                "mypy",
                "isort"
            ]
        }
        
    def validate_environment(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate development environment
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check Python version
        min_version = self.requirements["python"]["min_version"]
        current_version = self.requirements["python"]["current"]
        
        if current_version < min_version:
            errors.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"found {current_version[0]}.{current_version[1]}"
            )
        
        # Check required packages
        for package in self.requirements["packages"]:
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                errors.append(f"Required package missing: {package}")
                
        # Check optional packages
        for package in self.requirements["optional_packages"]:
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                warnings.append(f"Optional package missing: {package}")
                
        # Check framework components
        try:
            from . import IntegratedTestRunner, AssertionFramework, MockFramework
        except ImportError as e:
            errors.append(f"Framework component missing: {e}")
            
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
        
    def generate_environment_report(self) -> Dict[str, Any]:
        """Generate comprehensive environment report"""
        
        is_valid, errors, warnings = self.validate_environment()
        
        report = {
            "environment_status": "valid" if is_valid else "invalid",
            "python_version": {
                "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "required": f"{self.requirements['python']['min_version'][0]}.{self.requirements['python']['min_version'][1]}+",
                "valid": sys.version_info[:2] >= self.requirements["python"]["min_version"]
            },
            "packages": self._check_packages(),
            "errors": errors,
            "warnings": warnings,
            "suggestions": self._generate_suggestions(errors, warnings)
        }
        
        return report
        
    def _check_packages(self) -> Dict[str, Dict[str, Any]]:
        """Check package availability and versions"""
        
        package_info = {}
        
        all_packages = self.requirements["packages"] + self.requirements["optional_packages"]
        
        for package in all_packages:
            try:
                module = importlib.import_module(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                package_info[package] = {
                    "available": True,
                    "version": version,
                    "required": package in self.requirements["packages"]
                }
            except ImportError:
                package_info[package] = {
                    "available": False,
                    "version": None,
                    "required": package in self.requirements["packages"]
                }
                
        return package_info
        
    def _generate_suggestions(self, errors: List[str], warnings: List[str]) -> List[str]:
        """Generate suggestions based on errors and warnings"""
        
        suggestions = []
        
        if any("Python" in error for error in errors):
            suggestions.append("Upgrade Python to a supported version (3.8+)")
            
        missing_required = [pkg for pkg in self.requirements["packages"] 
                          if any(pkg in error for error in errors)]
        if missing_required:
            suggestions.append(f"Install required packages: pip install {' '.join(missing_required)}")
            
        missing_optional = [pkg for pkg in self.requirements["optional_packages"]
                          if any(pkg in warning for warning in warnings)]
        if missing_optional:
            suggestions.append(f"Install development tools: pip install {' '.join(missing_optional)}")
            
        if any("Framework component" in error for error in errors):
            suggestions.append("Reinstall the test-workflow framework: pip install -e .")
            
        return suggestions


class ProjectConfiguration:
    """Manages project configuration and setup"""
    
    def __init__(self, project_path: Path = None):
        self.project_path = project_path or Path.cwd()
        self.config_files = [
            "pyproject.toml",
            "setup.py", 
            "requirements.txt",
            "pytest.ini",
            ".gitignore"
        ]
        
    def validate_project_structure(self) -> Tuple[bool, List[str], List[str]]:
        """Validate project structure for test-workflow usage"""
        
        errors = []
        warnings = []
        
        # Check for basic project files
        essential_files = ["requirements.txt"]
        for file_name in essential_files:
            file_path = self.project_path / file_name
            if not file_path.exists():
                errors.append(f"Missing essential file: {file_name}")
                
        # Check for recommended files
        recommended_files = ["pytest.ini", ".gitignore"]
        for file_name in recommended_files:
            file_path = self.project_path / file_name
            if not file_path.exists():
                warnings.append(f"Recommended file missing: {file_name}")
                
        # Check directory structure
        recommended_dirs = ["tests", "src"]
        for dir_name in recommended_dirs:
            dir_path = self.project_path / dir_name
            if not dir_path.exists():
                warnings.append(f"Recommended directory missing: {dir_name}")
                
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
        
    def create_default_config(self, config_type: str = "minimal") -> Dict[str, str]:
        """Create default configuration files"""
        
        configs = {}
        
        if config_type in ["minimal", "full"]:
            # Create pytest.ini
            pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --cov=src
    --cov-report=term
    --cov-report=html
    --html=test_report.html
    --self-contained-html
asyncio_mode = auto
"""
            configs["pytest.ini"] = pytest_config
            
            # Create basic requirements.txt
            requirements = """# Test Workflow Framework
test-workflow>=1.0.0

# Testing dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-html>=3.1.0
"""
            
            if config_type == "full":
                requirements += """
# Development dependencies
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
isort>=5.12.0
"""
                
            configs["requirements.txt"] = requirements
            
            # Create .gitignore
            gitignore = """.pytest_cache/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.tox/
.coverage
htmlcov/
.nyc_output/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.env
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/
test_report.html
results/
temp/
*.log
"""
            configs[".gitignore"] = gitignore
            
        return configs
        
    def write_config_files(self, configs: Dict[str, str], overwrite: bool = False) -> List[str]:
        """Write configuration files to project"""
        
        written_files = []
        
        for filename, content in configs.items():
            file_path = self.project_path / filename
            
            if file_path.exists() and not overwrite:
                logger.info(f"Skipping existing file: {filename}")
                continue
                
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                written_files.append(filename)
                logger.info(f"Created configuration file: {filename}")
            except Exception as e:
                logger.error(f"Failed to create {filename}: {e}")
                
        return written_files


class ErrorDiagnostics:
    """Provides diagnostic information for common errors"""
    
    def __init__(self):
        self.diagnostics = {
            "import_error": self._diagnose_import_error,
            "assertion_error": self._diagnose_assertion_error,
            "mock_error": self._diagnose_mock_error,
            "context_error": self._diagnose_context_error,
            "async_error": self._diagnose_async_error
        }
        
    def diagnose_error(self, error: Exception, context: str = None) -> Dict[str, Any]:
        """Diagnose an error and provide helpful information"""
        
        error_type = type(error).__name__.lower()
        error_message = str(error)
        
        diagnosis = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "suggestions": [],
            "documentation_links": [],
            "code_examples": []
        }
        
        # Try specific diagnostics
        for diagnostic_key, diagnostic_func in self.diagnostics.items():
            if diagnostic_key in error_type or diagnostic_key in error_message.lower():
                specific_diagnosis = diagnostic_func(error, context)
                diagnosis.update(specific_diagnosis)
                break
        else:
            # Generic diagnosis
            diagnosis.update(self._diagnose_generic_error(error, context))
            
        return diagnosis
        
    def _diagnose_import_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Diagnose import-related errors"""
        
        return {
            "suggestions": [
                "Check if the package is installed: pip list | grep test-workflow",
                "Install missing dependencies: pip install -r requirements.txt",
                "Verify Python path includes the project directory",
                "Check for circular imports in your modules"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/installation",
                "https://test-workflow.readthedocs.io/troubleshooting#import-errors"
            ],
            "code_examples": [
                "# Correct import pattern\nfrom test_workflow.core import create_test_framework"
            ]
        }
        
    def _diagnose_assertion_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Diagnose assertion-related errors"""
        
        return {
            "suggestions": [
                "Check assertion method signature and parameters",
                "Verify expected vs actual values are correct types",
                "Use assertions.that() for fluent interface",
                "Add descriptive messages to assertions"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/assertions",
                "https://test-workflow.readthedocs.io/examples#assertion-examples"
            ],
            "code_examples": [
                "# Good assertion with message\nassertions.equal(actual, expected, 'User ID should match')"
            ]
        }
        
    def _diagnose_mock_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Diagnose mock-related errors"""
        
        return {
            "suggestions": [
                "Ensure mock is created before use: mocks.create_mock('name')",
                "Check mock configuration parameters",
                "Verify mock target exists and is accessible",
                "Use spies for partial mocking of real objects"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/mocking",
                "https://test-workflow.readthedocs.io/examples#mocking-examples"
            ],
            "code_examples": [
                "# Create and configure mock\napi_mock = mocks.create_mock('api_client')\napi_mock.get.return_value = {'data': 'test'}"
            ]
        }
        
    def _diagnose_context_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Diagnose context-related errors"""
        
        return {
            "suggestions": [
                "Check that context key exists before access: context.has('key')",
                "Use context.get('key', default) for safe access",
                "Verify context setup in suite.setup function",
                "Ensure proper context cleanup"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/context",
                "https://test-workflow.readthedocs.io/examples#context-examples"
            ],
            "code_examples": [
                "# Safe context access\nvalue = context.get('key', default_value)"
            ]
        }
        
    def _diagnose_async_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Diagnose async-related errors"""
        
        return {
            "suggestions": [
                "Use async def for async test functions",
                "Ensure asyncio.run() in main execution",
                "Use await with async operations",
                "Create async mocks for async services"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/async-testing",
                "https://test-workflow.readthedocs.io/examples#async-examples"
            ],
            "code_examples": [
                "# Async test function\nasync def test_async_operation(assertions, mocks, context):\n    result = await some_async_function()\n    assertions.is_not_none(result)"
            ]
        }
        
    def _diagnose_generic_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Provide generic diagnosis for unknown errors"""
        
        return {
            "suggestions": [
                "Check the error message and traceback carefully",
                "Verify all required dependencies are installed",
                "Review test function signatures and parameters",
                "Check framework documentation for similar issues"
            ],
            "documentation_links": [
                "https://test-workflow.readthedocs.io/troubleshooting",
                "https://test-workflow.readthedocs.io/faq"
            ],
            "code_examples": [
                "# Basic test function structure\ndef test_example(assertions, mocks, context):\n    # Test implementation"
            ]
        }


def setup_development_environment(
    project_path: Path = None,
    config_type: str = "minimal",
    install_deps: bool = False
) -> Dict[str, Any]:
    """
    Setup development environment for test-workflow
    
    Args:
        project_path: Path to project directory
        config_type: Type of configuration ('minimal' or 'full')
        install_deps: Whether to install dependencies
        
    Returns:
        Setup results dictionary
    """
    
    project_path = project_path or Path.cwd()
    
    print(f"🔧 Setting up Test Workflow Framework development environment")
    print(f"Project path: {project_path}")
    print(f"Configuration type: {config_type}")
    
    # Validate environment
    env = DevelopmentEnvironment()
    is_valid, errors, warnings = env.validate_environment()
    
    if errors:
        print(f"❌ Environment validation failed:")
        for error in errors:
            print(f"  • {error}")
        return {"status": "failed", "errors": errors}
        
    if warnings:
        print(f"⚠️ Environment warnings:")
        for warning in warnings:
            print(f"  • {warning}")
            
    # Setup project configuration
    config = ProjectConfiguration(project_path)
    configs = config.create_default_config(config_type)
    written_files = config.write_config_files(configs)
    
    print(f"📄 Created configuration files:")
    for filename in written_files:
        print(f"  • {filename}")
        
    # Install dependencies if requested
    if install_deps:
        print("📦 Installing dependencies...")
        try:
            requirements_file = project_path / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Dependencies installed successfully")
            else:
                print("⚠️ No requirements.txt found for dependency installation")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return {"status": "failed", "errors": [str(e)]}
            
    print("🎉 Development environment setup completed!")
    
    return {
        "status": "success",
        "project_path": str(project_path),
        "config_type": config_type,
        "files_created": written_files,
        "dependencies_installed": install_deps,
        "warnings": warnings
    }


def validate_environment() -> Dict[str, Any]:
    """
    Validate current environment for test-workflow usage
    
    Returns:
        Validation results dictionary
    """
    
    env = DevelopmentEnvironment()
    report = env.generate_environment_report()
    
    print("🔍 Validating Test Workflow Framework environment...")
    print(f"Status: {'✅ Valid' if report['environment_status'] == 'valid' else '❌ Invalid'}")
    
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"  • {error}")
            
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  • {warning}")
            
    if report["suggestions"]:
        print("💡 Suggestions:")
        for suggestion in report["suggestions"]:
            print(f"  • {suggestion}")
            
    return report


def diagnose_error(error: Exception, context: str = None) -> Dict[str, Any]:
    """
    Diagnose error and provide helpful information
    
    Args:
        error: Exception to diagnose
        context: Additional context information
        
    Returns:
        Diagnostic information dictionary
    """
    
    diagnostics = ErrorDiagnostics()
    diagnosis = diagnostics.diagnose_error(error, context)
    
    print(f"🔍 Diagnosing error: {type(error).__name__}")
    print(f"Message: {diagnosis['error_message']}")
    
    if diagnosis["suggestions"]:
        print("💡 Suggestions:")
        for suggestion in diagnosis["suggestions"]:
            print(f"  • {suggestion}")
            
    if diagnosis["code_examples"]:
        print("📝 Code examples:")
        for example in diagnosis["code_examples"]:
            print(f"  {example}")
            
    return diagnosis