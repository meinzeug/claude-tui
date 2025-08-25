"""
Project-related test fixtures and factories.
"""

import pytest
from faker import Faker
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

fake = Faker()


class ProjectTemplate(Enum):
    """Available project templates."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    REACT = "react"
    FASTAPI = "fastapi"
    CLI = "cli"


class ProjectStatus(Enum):
    """Project status values."""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class MockProject:
    """Mock project data structure."""
    id: str
    name: str
    description: str
    template: str
    status: str
    author: str
    email: str
    version: str
    created_at: str
    updated_at: str
    path: Path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "status": self.status,
            "author": self.author,
            "email": self.email,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path": str(self.path)
        }


class ProjectFactory:
    """Factory for creating test project data."""
    
    @staticmethod
    def create_project_data(**overrides) -> Dict[str, Any]:
        """Create project data dictionary."""
        defaults = {
            "name": fake.slug(),
            "description": fake.text(max_nb_chars=200),
            "template": ProjectTemplate.PYTHON.value,
            "status": ProjectStatus.INITIALIZED.value,
            "author": fake.name(),
            "email": fake.email(),
            "version": "0.1.0"
        }
        defaults.update(overrides)
        return defaults
    
    @staticmethod
    def create_mock_project(project_dir: Path, **overrides) -> MockProject:
        """Create mock project instance."""
        data = ProjectFactory.create_project_data(**overrides)
        return MockProject(
            id=fake.uuid4(),
            path=project_dir,
            created_at=fake.iso8601(),
            updated_at=fake.iso8601(),
            **data
        )
    
    @staticmethod
    def create_project_structure(base_dir: Path, template: str = "python") -> Dict[str, Path]:
        """Create project directory structure."""
        structure = {
            "root": base_dir,
            "src": base_dir / "src",
            "tests": base_dir / "tests",
            "docs": base_dir / "docs",
            "config": base_dir / "config"
        }
        
        # Create directories
        for path in structure.values():
            path.mkdir(exist_ok=True)
        
        # Create template-specific files
        if template == "python":
            structure["src_init"] = structure["src"] / "__init__.py"
            structure["src_init"].touch()
            structure["main"] = structure["src"] / "main.py"
            structure["main"].write_text("def main():\n    pass\n")
            structure["requirements"] = base_dir / "requirements.txt"
            structure["requirements"].write_text("# Project dependencies\n")
            
        elif template == "fastapi":
            structure["app"] = structure["src"] / "app.py"
            structure["app"].write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
""")
            structure["requirements"].write_text("fastapi>=0.100.0\nuvicorn>=0.23.0\n")
        
        return structure


@pytest.fixture
def project_factory():
    """Provide project factory for tests."""
    return ProjectFactory


@pytest.fixture
def sample_project_data():
    """Generate sample project data."""
    return ProjectFactory.create_project_data()


@pytest.fixture
def python_project_data():
    """Generate Python project data."""
    return ProjectFactory.create_project_data(
        template=ProjectTemplate.PYTHON.value,
        name="test-python-project"
    )


@pytest.fixture
def fastapi_project_data():
    """Generate FastAPI project data."""
    return ProjectFactory.create_project_data(
        template=ProjectTemplate.FASTAPI.value,
        name="test-fastapi-project"
    )


@pytest.fixture
def mock_project_with_structure(tmp_path):
    """Create mock project with full directory structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create structure
    structure = ProjectFactory.create_project_structure(project_dir)
    
    # Create mock project
    project = ProjectFactory.create_mock_project(project_dir)
    
    return {
        "project": project,
        "structure": structure,
        "path": project_dir
    }


@pytest.fixture
def multiple_projects(tmp_path):
    """Create multiple test projects."""
    projects = []
    
    for i in range(3):
        project_dir = tmp_path / f"project_{i}"
        project_dir.mkdir()
        
        template = [ProjectTemplate.PYTHON.value, ProjectTemplate.FASTAPI.value, ProjectTemplate.CLI.value][i]
        structure = ProjectFactory.create_project_structure(project_dir, template)
        
        project = ProjectFactory.create_mock_project(
            project_dir,
            name=f"test-project-{i}",
            template=template,
            status=ProjectStatus.IN_PROGRESS.value if i == 0 else ProjectStatus.INITIALIZED.value
        )
        
        projects.append({
            "project": project,
            "structure": structure,
            "path": project_dir
        })
    
    return projects


class ProjectTestHelper:
    """Helper methods for project testing."""
    
    @staticmethod
    def assert_valid_project_data(data: Dict[str, Any]):
        """Assert project data has required fields."""
        required_fields = ["name", "template", "status", "author", "email"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert len(data["name"]) > 0, "Project name cannot be empty"
        assert "@" in data["email"], "Invalid email format"
        assert data["template"] in [t.value for t in ProjectTemplate], f"Invalid template: {data['template']}"
        assert data["status"] in [s.value for s in ProjectStatus], f"Invalid status: {data['status']}"
    
    @staticmethod
    def assert_project_structure_exists(structure: Dict[str, Path]):
        """Assert all required project directories exist."""
        required_dirs = ["root", "src", "tests", "docs"]
        for dir_name in required_dirs:
            assert dir_name in structure, f"Missing directory: {dir_name}"
            assert structure[dir_name].exists(), f"Directory does not exist: {structure[dir_name]}"
    
    @staticmethod
    def create_sample_code_files(src_dir: Path, template: str = "python"):
        """Create sample code files for testing."""
        if template == "python":
            # Create main module
            main_file = src_dir / "main.py"
            main_file.write_text("""
def main():
    \"\"\"Main application entry point.\"\"\"
    print("Hello, claude-tiu!")
    
    # TODO: implement CLI interface
    pass

if __name__ == "__main__":
    main()
""")
            
            # Create utility module
            utils_file = src_dir / "utils.py"
            utils_file.write_text("""
def helper_function():
    \"\"\"Helper function that's complete.\"\"\"
    return "Helper result"

def incomplete_function():
    \"\"\"Function with placeholder.\"\"\"
    raise NotImplementedError("To be implemented")
""")
            
        elif template == "fastapi":
            # Create FastAPI app
            app_file = src_dir / "app.py"
            app_file.write_text("""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Test API", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/status")
async def status():
    # TODO: implement health check
    pass

@app.post("/items/")
async def create_item(item: dict):
    # implement later
    return JSONResponse({"status": "placeholder"})
""")


@pytest.fixture
def project_helper():
    """Provide project test helper."""
    return ProjectTestHelper