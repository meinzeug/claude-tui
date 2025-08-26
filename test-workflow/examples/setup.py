#!/usr/bin/env python3
"""
Setup script for Test Workflow Framework examples
Creates necessary directories and configuration files
"""

import os
import sys
from pathlib import Path
import json


def create_directory_structure():
    """Create the necessary directory structure for examples"""
    
    base_dir = Path(__file__).parent
    
    # Create results directories
    directories = [
        "results",
        "results/basic",
        "results/advanced", 
        "results/performance",
        "results/integration",
        "temp",
        "fixtures",
        "data"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
        
        # Create .gitkeep for empty directories
        gitkeep_file = dir_path / ".gitkeep"
        if not any(dir_path.iterdir()):  # If directory is empty
            gitkeep_file.touch()


def create_requirements_file():
    """Create requirements.txt for examples"""
    
    base_dir = Path(__file__).parent
    requirements_file = base_dir / "requirements.txt"
    
    requirements = [
        "# Test Workflow Framework Examples Requirements",
        "",
        "# Core framework (adjust version as needed)", 
        "test-workflow>=1.0.0",
        "",
        "# Additional dependencies for examples",
        "asyncio",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-html>=3.1.0",
        "pytest-benchmark>=4.0.0",
        "",
        "# For advanced examples",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.5.0",
        "",
        "# Development tools",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "isort>=5.12.0"
    ]
    
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"📄 Created requirements file: {requirements_file}")


def create_config_file():
    """Create configuration file for examples"""
    
    base_dir = Path(__file__).parent
    config_file = base_dir / "config.json"
    
    config = {
        "framework": {
            "name": "test-workflow",
            "version": "1.0.0",
            "description": "Integrated testing framework for SPARC methodology"
        },
        "reporting": {
            "default_formats": ["console", "json", "html"],
            "output_directory": "results",
            "include_coverage": True,
            "include_timing": True,
            "include_stack_traces": True
        },
        "execution": {
            "parallel_execution": True,
            "max_workers": 4,
            "timeout_seconds": 300,
            "retry_failed_tests": False
        },
        "examples": {
            "basic": {
                "enabled": True,
                "output_dir": "results/basic"
            },
            "advanced": {
                "enabled": True,
                "output_dir": "results/advanced"
            },
            "performance": {
                "enabled": True,
                "output_dir": "results/performance"
            },
            "integration": {
                "enabled": True,
                "output_dir": "results/integration"
            }
        },
        "test_data": {
            "fixtures_directory": "fixtures",
            "temp_directory": "temp",
            "cleanup_after_tests": True
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Created config file: {config_file}")


def create_example_test_data():
    """Create example test data files"""
    
    base_dir = Path(__file__).parent
    fixtures_dir = base_dir / "fixtures"
    
    # Sample user data
    users_file = fixtures_dir / "users.json"
    users_data = [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "active": True,
            "created_at": "2023-01-01T00:00:00Z"
        },
        {
            "id": 2,
            "name": "Jane Smith", 
            "email": "jane@example.com",
            "active": True,
            "created_at": "2023-01-02T00:00:00Z"
        },
        {
            "id": 3,
            "name": "Bob Johnson",
            "email": "bob@example.com", 
            "active": False,
            "created_at": "2023-01-03T00:00:00Z"
        }
    ]
    
    with open(users_file, 'w') as f:
        json.dump(users_data, f, indent=2)
    
    print(f"📊 Created test data: {users_file}")
    
    # Sample configuration data
    config_data_file = fixtures_dir / "test_config.json"
    test_config = {
        "database": {
            "url": "sqlite:///test.db",
            "pool_size": 5,
            "timeout": 30
        },
        "api": {
            "base_url": "https://api.example.com",
            "timeout": 10,
            "retries": 3
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    with open(config_data_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"⚙️ Created test config: {config_data_file}")


def create_makefile():
    """Create Makefile for convenient example execution"""
    
    base_dir = Path(__file__).parent
    makefile = base_dir / "Makefile"
    
    makefile_content = """# Test Workflow Framework Examples Makefile

.PHONY: help setup clean test test-basic test-advanced test-performance test-integration all

help:
	@echo "Test Workflow Framework Examples"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup           - Setup examples environment"
	@echo "  clean           - Clean generated files"
	@echo "  test            - Run all example tests"
	@echo "  test-basic      - Run basic usage examples"
	@echo "  test-advanced   - Run advanced usage examples"
	@echo "  test-performance- Run performance examples"
	@echo "  test-integration- Run integration examples"
	@echo "  all             - Setup and run all tests"

setup:
	@echo "🔧 Setting up examples environment..."
	python setup.py
	pip install -r requirements.txt

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf results/*/
	rm -rf temp/*
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test-basic:
	@echo "🚀 Running basic usage examples..."
	python basic_usage.py

test-advanced:
	@echo "🚀 Running advanced usage examples..."
	python advanced_usage.py

test-performance:
	@echo "📊 Running performance examples..."
	python -c "import asyncio; from advanced_usage import run_advanced_examples; asyncio.run(run_advanced_examples())" --filter-tags performance

test-integration:
	@echo "🔗 Running integration tests..."
	python -m pytest tests/test_integration.py -v

test: test-basic test-advanced test-integration
	@echo "✅ All example tests completed!"

all: setup test
	@echo "🎉 Setup and testing completed successfully!"

# Development targets
install-dev:
	pip install -e ../../  # Install framework in development mode
	pip install -r requirements.txt

lint:
	flake8 *.py
	black --check *.py
	mypy *.py --ignore-missing-imports

format:
	black *.py
	isort *.py

# CI/CD targets
ci-test:
	python basic_usage.py
	python advanced_usage.py
	python -m pytest tests/ -v --junitxml=junit.xml --html=report.html

# Documentation targets
docs:
	@echo "📚 Example documentation:"
	@echo "  getting_started.md - Getting started guide"
	@echo "  basic_usage.py     - Basic usage examples"
	@echo "  advanced_usage.py  - Advanced usage examples"
	@echo ""
	@echo "📊 Generated reports available in results/ directories"
"""
    
    with open(makefile, 'w') as f:
        f.write(makefile_content)
    
    print(f"🔨 Created Makefile: {makefile}")


def create_readme():
    """Create README for examples directory"""
    
    base_dir = Path(__file__).parent
    readme_file = base_dir / "README.md"
    
    readme_content = """# Test Workflow Framework Examples

This directory contains comprehensive examples demonstrating the Test Workflow Framework capabilities.

## 🚀 Quick Start

1. **Setup the environment:**
   ```bash
   make setup
   # or
   python setup.py
   pip install -r requirements.txt
   ```

2. **Run basic examples:**
   ```bash
   make test-basic
   # or
   python basic_usage.py
   ```

3. **Run advanced examples:**
   ```bash
   make test-advanced  
   # or
   python advanced_usage.py
   ```

## 📁 Directory Structure

```
examples/
├── README.md                 # This file
├── getting_started.md        # Comprehensive getting started guide
├── setup.py                  # Environment setup script
├── requirements.txt          # Python dependencies
├── config.json              # Example configuration
├── Makefile                 # Convenient build targets
├── basic_usage.py           # Basic usage examples
├── advanced_usage.py        # Advanced usage examples
├── fixtures/                # Test data files
│   ├── users.json          # Sample user data
│   └── test_config.json    # Sample configuration
├── results/                 # Generated test reports
│   ├── basic/              # Basic example results
│   ├── advanced/           # Advanced example results
│   ├── performance/        # Performance test results
│   └── integration/        # Integration test results
├── tests/                  # Integration tests
│   └── test_integration.py # Framework integration tests
└── temp/                   # Temporary files
```

## 📋 Available Examples

### Basic Usage (`basic_usage.py`)
- Simple test creation
- Basic assertions
- Mock usage
- Context management
- Error handling
- Console and file reporting

### Advanced Usage (`advanced_usage.py`)
- Complex integration testing
- Performance benchmarking
- Parallel test execution
- Advanced mocking scenarios
- Context inheritance and snapshots
- Multiple report formats
- Error scenario testing
- Hook system usage

### Integration Tests (`tests/test_integration.py`)
- Component interaction testing
- End-to-end framework validation
- Error handling verification
- Performance validation

## 🔧 Make Targets

- `make help` - Show available targets
- `make setup` - Setup environment
- `make test` - Run all examples
- `make test-basic` - Run basic examples only
- `make test-advanced` - Run advanced examples only
- `make test-integration` - Run integration tests
- `make clean` - Clean generated files
- `make all` - Setup and run everything

## 📊 Generated Reports

After running examples, check the `results/` directory for:
- **Console Output**: Displayed in terminal
- **JSON Reports**: Machine-readable test results
- **HTML Reports**: Interactive web reports
- **JUnit XML**: CI/CD compatible format
- **Markdown Reports**: Documentation-friendly format

## 🎯 Key Learning Points

1. **Framework Creation**: Different ways to set up the test framework
2. **Test Organization**: How to structure test suites and functions
3. **Assertions**: Comprehensive assertion patterns and fluent interface
4. **Mocking**: Advanced mocking, spying, and stubbing techniques
5. **Context Management**: Shared data, fixtures, and cleanup
6. **Async Testing**: Patterns for testing async code
7. **Reporting**: Multiple output formats and configuration
8. **Integration**: How all components work together
9. **CI/CD**: Ready-to-use pipeline configurations
10. **Best Practices**: Recommended patterns and approaches

## 🆘 Troubleshooting

**Dependencies not installing:**
```bash
# Update pip and try again
pip install --upgrade pip
pip install -r requirements.txt
```

**Import errors:**
```bash
# Install framework in development mode
pip install -e ../../
```

**Permission errors on cleanup:**
```bash
# On Windows, you might need to run as administrator
# On Unix systems, check file permissions
chmod -R 755 results/
```

## 🎉 Next Steps

1. Study the `getting_started.md` guide
2. Run the basic examples to understand core concepts
3. Explore advanced examples for complex scenarios
4. Try creating your own test suites
5. Integrate with your existing projects
6. Set up CI/CD using provided configurations

Happy testing! 🧪✨
"""
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"📚 Created README: {readme_file}")


def main():
    """Main setup function"""
    
    print("🚀 Setting up Test Workflow Framework Examples")
    print("=" * 50)
    
    try:
        create_directory_structure()
        create_requirements_file()
        create_config_file()
        create_example_test_data()
        create_makefile()
        create_readme()
        
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run basic examples: python basic_usage.py")
        print("3. Run advanced examples: python advanced_usage.py")
        print("4. Check getting_started.md for comprehensive guide")
        print("5. Use 'make help' for available commands")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()