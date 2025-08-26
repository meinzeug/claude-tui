#!/usr/bin/env python3
"""
Claude-TUI Documentation Auto-Generation Pipeline
=================================================

This script automatically generates comprehensive documentation from the codebase,
including API references, code examples, and interactive tutorials.

Features:
- Auto-generated API documentation from OpenAPI specs
- Code example extraction and validation
- Interactive tutorial generation
- Multi-format output (HTML, PDF, Markdown)
- Real-time documentation updates
"""

import os
import sys
import json
import yaml
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import importlib.util
import inspect
import ast
import re
from datetime import datetime

# Third-party imports
import click
import jinja2
from pydantic import BaseModel
import markdown
from markdown.extensions import codehilite, toc
from mkdocs.config import load_config
from mkdocs.commands.build import build


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    
    # Source paths
    source_dir: Path = Path("src")
    docs_dir: Path = Path("docs")
    api_spec_path: Path = Path("docs/openapi-specification.yaml")
    
    # Output configuration
    output_formats: List[str] = None
    build_dir: Path = Path("docs/_build")
    
    # Generation options
    include_private: bool = False
    include_tests: bool = True
    generate_examples: bool = True
    validate_code: bool = True
    
    # Templates
    template_dir: Path = Path("docs/_templates")
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["html", "markdown"]
            

class CodeAnalyzer:
    """Analyzes Python code to extract documentation information."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a Python module and extract documentation info."""
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            module_info = {
                "path": str(module_path),
                "name": module_path.stem,
                "docstring": ast.get_docstring(tree),
                "classes": [],
                "functions": [],
                "imports": [],
                "examples": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    module_info["classes"].append(class_info)
                    
                elif isinstance(node, ast.FunctionDef):
                    # Skip private functions unless configured
                    if not node.name.startswith('_') or self.config.include_private:
                        func_info = self._analyze_function(node)
                        module_info["functions"].append(func_info)
                        
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    module_info["imports"].append(import_info)
            
            # Extract code examples from docstrings and comments
            examples = self._extract_examples(content)
            module_info["examples"].extend(examples)
            
            return module_info
            
        except Exception as e:
            print(f"Error analyzing {module_path}: {e}")
            return {}
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(base) for base in node.bases],
            "methods": [],
            "properties": [],
            "line_number": node.lineno
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_') or self.config.include_private:
                    method_info = self._analyze_function(item, is_method=True)
                    
                    # Check if it's a property
                    if any(isinstance(dec, ast.Name) and dec.id == 'property' 
                           for dec in item.decorator_list):
                        class_info["properties"].append(method_info)
                    else:
                        class_info["methods"].append(method_info)
        
        return class_info
    
    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Analyze a function definition."""
        
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": [],
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "is_method": is_method,
            "line_number": node.lineno
        }
        
        # Analyze arguments
        for arg in node.args.args:
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_annotation(arg.annotation) if arg.annotation else None
            }
            func_info["args"].append(arg_info)
        
        # Analyze return type
        if node.returns:
            func_info["returns"] = self._get_annotation(node.returns)
        
        # Analyze decorators
        for decorator in node.decorator_list:
            decorator_name = self._get_name(decorator)
            func_info["decorators"].append(decorator_name)
        
        return func_info
    
    def _analyze_import(self, node: ast.Import | ast.ImportFrom) -> Dict[str, Any]:
        """Analyze an import statement."""
        
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names],
                "line_number": node.lineno
            }
        else:
            return {
                "type": "from_import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "level": node.level,
                "line_number": node.lineno
            }
    
    def _extract_examples(self, content: str) -> List[Dict[str, Any]]:
        """Extract code examples from docstrings and comments."""
        
        examples = []
        
        # Extract examples from docstrings (```python blocks)
        docstring_examples = re.findall(
            r'""".*?```python\n(.*?)\n```.*?"""',
            content,
            re.DOTALL | re.MULTILINE
        )
        
        for i, example in enumerate(docstring_examples):
            examples.append({
                "type": "docstring",
                "code": example.strip(),
                "index": i,
                "validated": False
            })
        
        # Extract examples from comments (# Example: blocks)
        comment_examples = re.findall(
            r'# Example:.*?\n((?:\s*#.*\n)*)',
            content,
            re.MULTILINE
        )
        
        for i, example in enumerate(comment_examples):
            # Clean up comment markers
            cleaned = re.sub(r'^\s*#\s?', '', example, flags=re.MULTILINE)
            if cleaned.strip():
                examples.append({
                    "type": "comment",
                    "code": cleaned.strip(),
                    "index": i,
                    "validated": False
                })
        
        return examples
    
    def _get_name(self, node: ast.expr) -> str:
        """Get the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return ast.unparse(node)
    
    def _get_annotation(self, node: ast.expr) -> str:
        """Get type annotation as string."""
        return ast.unparse(node)


class APIDocumentationGenerator:
    """Generates API documentation from OpenAPI specifications."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        
    def generate_api_docs(self) -> Dict[str, Any]:
        """Generate API documentation from OpenAPI spec."""
        
        if not self.config.api_spec_path.exists():
            print(f"OpenAPI spec not found at {self.config.api_spec_path}")
            return {}
        
        with open(self.config.api_spec_path, 'r') as f:
            spec = yaml.safe_load(f)
        
        api_docs = {
            "info": spec.get("info", {}),
            "servers": spec.get("servers", []),
            "paths": {},
            "components": spec.get("components", {}),
            "security": spec.get("security", [])
        }
        
        # Process paths
        for path, path_info in spec.get("paths", {}).items():
            api_docs["paths"][path] = {}
            
            for method, method_info in path_info.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]:
                    api_docs["paths"][path][method.upper()] = {
                        "summary": method_info.get("summary", ""),
                        "description": method_info.get("description", ""),
                        "parameters": method_info.get("parameters", []),
                        "requestBody": method_info.get("requestBody"),
                        "responses": method_info.get("responses", {}),
                        "tags": method_info.get("tags", []),
                        "security": method_info.get("security", [])
                    }
        
        return api_docs


class ExampleValidator:
    """Validates code examples found in documentation."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        
    async def validate_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate code examples."""
        
        if not self.config.validate_code:
            return examples
        
        validated_examples = []
        
        for example in examples:
            try:
                # Try to parse the code
                ast.parse(example["code"])
                
                # Try to run static analysis
                validation_result = await self._run_static_analysis(example["code"])
                
                example["validated"] = True
                example["validation_result"] = validation_result
                
            except SyntaxError as e:
                example["validated"] = False
                example["validation_error"] = f"Syntax error: {e}"
                
            except Exception as e:
                example["validated"] = False
                example["validation_error"] = f"Validation error: {e}"
            
            validated_examples.append(example)
        
        return validated_examples
    
    async def _run_static_analysis(self, code: str) -> Dict[str, Any]:
        """Run static analysis on code example."""
        
        # This would integrate with tools like mypy, pylint, etc.
        # For now, just basic AST validation
        
        tree = ast.parse(code)
        
        analysis = {
            "syntax_valid": True,
            "imports_used": [],
            "functions_defined": [],
            "classes_defined": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                analysis["imports_used"].append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                analysis["functions_defined"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis["classes_defined"].append(node.name)
        
        return analysis


class DocumentationRenderer:
    """Renders documentation using Jinja2 templates."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(config.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters
        self.jinja_env.filters['markdown'] = self._markdown_filter
        self.jinja_env.filters['code_highlight'] = self._code_highlight_filter
    
    def render_module_docs(self, module_info: Dict[str, Any]) -> str:
        """Render documentation for a module."""
        
        template = self.jinja_env.get_template('module.md.j2')
        return template.render(module=module_info, timestamp=datetime.now())
    
    def render_api_docs(self, api_info: Dict[str, Any]) -> str:
        """Render API documentation."""
        
        template = self.jinja_env.get_template('api.md.j2')
        return template.render(api=api_info, timestamp=datetime.now())
    
    def render_tutorial(self, tutorial_info: Dict[str, Any]) -> str:
        """Render interactive tutorial."""
        
        template = self.jinja_env.get_template('tutorial.md.j2')
        return template.render(tutorial=tutorial_info, timestamp=datetime.now())
    
    def _markdown_filter(self, text: str) -> str:
        """Convert markdown to HTML."""
        md = markdown.Markdown(extensions=['codehilite', 'toc'])
        return md.convert(text)
    
    def _code_highlight_filter(self, code: str, language: str = 'python') -> str:
        """Add syntax highlighting to code."""
        return f"```{language}\n{code}\n```"


class TutorialGenerator:
    """Generates interactive tutorials from code examples."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        
    def generate_tutorials(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate interactive tutorials from modules."""
        
        tutorials = []
        
        # Basic usage tutorial
        basic_tutorial = {
            "title": "Getting Started with Claude-TUI",
            "description": "Learn the basics of using Claude-TUI's intelligent development brain",
            "sections": [
                {
                    "title": "Installation",
                    "content": "## Installing Claude-TUI\n\n```bash\npip install claude-tui\n```",
                    "code_example": "pip install claude-tui",
                    "expected_output": "Successfully installed claude-tui"
                },
                {
                    "title": "First Project",
                    "content": "## Creating Your First Project",
                    "code_example": """
from claude_tui import ProjectManager

# Create a project manager
manager = ProjectManager()

# Create a new project
project = await manager.create_project({
    "name": "my-first-app",
    "template": "python-package"
})

print(f"Project created: {project.name}")
""",
                    "expected_output": "Project created: my-first-app"
                }
            ]
        }
        tutorials.append(basic_tutorial)
        
        # API tutorial
        api_tutorial = {
            "title": "Working with the API",
            "description": "Learn how to use Claude-TUI's REST API",
            "sections": [
                {
                    "title": "Authentication",
                    "content": "## API Authentication",
                    "code_example": """
import requests

# Login to get JWT token
response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    "email": "user@example.com",
    "password": "password123"
})

token = response.json()["data"]["access_token"]
print(f"Token: {token[:20]}...")
""",
                    "expected_output": "Token: eyJ0eXAiOiJKV1QiLCJ..."
                }
            ]
        }
        tutorials.append(api_tutorial)
        
        # Generate tutorials from module examples
        for module in modules:
            if module.get("examples"):
                tutorial = self._create_tutorial_from_module(module)
                if tutorial:
                    tutorials.append(tutorial)
        
        return tutorials
    
    def _create_tutorial_from_module(self, module: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a tutorial from a module's examples."""
        
        if not module.get("examples"):
            return None
        
        tutorial = {
            "title": f"Using {module['name']}",
            "description": module.get("docstring", f"Learn how to use the {module['name']} module"),
            "sections": []
        }
        
        for i, example in enumerate(module["examples"]):
            section = {
                "title": f"Example {i + 1}",
                "content": f"## Working with {module['name']}",
                "code_example": example["code"],
                "expected_output": "# Output will vary based on your setup"
            }
            tutorial["sections"].append(section)
        
        return tutorial if tutorial["sections"] else None


class DocumentationBuilder:
    """Main documentation builder that orchestrates all components."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.analyzer = CodeAnalyzer(config)
        self.api_generator = APIDocumentationGenerator(config)
        self.validator = ExampleValidator(config)
        self.renderer = DocumentationRenderer(config)
        self.tutorial_generator = TutorialGenerator(config)
        
    async def build_documentation(self):
        """Build complete documentation."""
        
        print("🚀 Starting documentation generation...")
        
        # Create output directory
        self.config.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze source code
        print("📊 Analyzing source code...")
        modules = await self._analyze_codebase()
        
        # Generate API documentation
        print("🌐 Generating API documentation...")
        api_docs = self.api_generator.generate_api_docs()
        
        # Validate examples
        print("✅ Validating code examples...")
        await self._validate_all_examples(modules)
        
        # Generate tutorials
        print("📚 Generating tutorials...")
        tutorials = self.tutorial_generator.generate_tutorials(modules)
        
        # Render documentation
        print("🎨 Rendering documentation...")
        await self._render_all_docs(modules, api_docs, tutorials)
        
        # Build static site
        if "html" in self.config.output_formats:
            print("🏗️ Building static site...")
            await self._build_static_site()
        
        print(f"✨ Documentation generated in {self.config.build_dir}")
    
    async def _analyze_codebase(self) -> List[Dict[str, Any]]:
        """Analyze entire codebase."""
        
        modules = []
        
        for py_file in self.config.source_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            # Skip test files unless configured
            if "test" in py_file.parts and not self.config.include_tests:
                continue
            
            module_info = self.analyzer.analyze_module(py_file)
            if module_info:
                modules.append(module_info)
        
        return modules
    
    async def _validate_all_examples(self, modules: List[Dict[str, Any]]):
        """Validate all code examples."""
        
        for module in modules:
            if module.get("examples"):
                validated = await self.validator.validate_examples(module["examples"])
                module["examples"] = validated
    
    async def _render_all_docs(
        self, 
        modules: List[Dict[str, Any]], 
        api_docs: Dict[str, Any], 
        tutorials: List[Dict[str, Any]]
    ):
        """Render all documentation."""
        
        # Render module documentation
        for module in modules:
            module_doc = self.renderer.render_module_docs(module)
            
            output_path = self.config.build_dir / "modules" / f"{module['name']}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(module_doc)
        
        # Render API documentation
        if api_docs:
            api_doc = self.renderer.render_api_docs(api_docs)
            
            with open(self.config.build_dir / "api-reference.md", 'w') as f:
                f.write(api_doc)
        
        # Render tutorials
        for i, tutorial in enumerate(tutorials):
            tutorial_doc = self.renderer.render_tutorial(tutorial)
            
            output_path = self.config.build_dir / "tutorials" / f"tutorial-{i+1}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(tutorial_doc)
    
    async def _build_static_site(self):
        """Build static HTML site using MkDocs."""
        
        # Create MkDocs config
        mkdocs_config = {
            'site_name': 'Claude-TUI Documentation',
            'site_description': 'Comprehensive documentation for Claude-TUI intelligent development brain',
            'docs_dir': str(self.config.build_dir),
            'site_dir': str(self.config.build_dir / 'site'),
            'theme': {
                'name': 'material',
                'palette': {
                    'scheme': 'slate',
                    'primary': 'blue',
                    'accent': 'blue'
                },
                'features': [
                    'navigation.tabs',
                    'navigation.sections',
                    'navigation.expand',
                    'navigation.top',
                    'search.highlight',
                    'content.code.annotate'
                ]
            },
            'markdown_extensions': [
                'codehilite',
                'admonition',
                'toc',
                'attr_list',
                'pymdownx.superfences',
                'pymdownx.tabbed'
            ],
            'nav': [
                {'Home': 'index.md'},
                {'User Guide': [
                    'user-guide/getting-started.md',
                    'user-guide/installation.md'
                ]},
                {'Developer Guide': [
                    'developer-guide/architecture-deep-dive.md',
                    'developer-guide/contributing.md'
                ]},
                {'API Reference': 'api-reference.md'},
                {'Tutorials': []},
                {'Operations': [
                    'operations/production-deployment.md'
                ]}
            ]
        }
        
        # Add tutorials to nav
        tutorial_files = list((self.config.build_dir / 'tutorials').glob('*.md'))
        if tutorial_files:
            tutorial_nav = []
            for tutorial_file in sorted(tutorial_files):
                title = tutorial_file.stem.replace('-', ' ').title()
                tutorial_nav.append({title: f'tutorials/{tutorial_file.name}'})
            mkdocs_config['nav'][4]['Tutorials'] = tutorial_nav
        
        # Write MkDocs config
        config_path = self.config.build_dir / 'mkdocs.yml'
        with open(config_path, 'w') as f:
            yaml.dump(mkdocs_config, f, default_flow_style=False)
        
        # Build site
        subprocess.run(['mkdocs', 'build', '-f', str(config_path)], check=True)


# CLI Interface
@click.group()
def cli():
    """Claude-TUI Documentation Generator."""
    pass


@cli.command()
@click.option('--source-dir', default='src', help='Source code directory')
@click.option('--docs-dir', default='docs', help='Documentation directory') 
@click.option('--output-dir', default='docs/_build', help='Output directory')
@click.option('--format', 'output_formats', multiple=True, default=['html', 'markdown'], help='Output formats')
@click.option('--validate/--no-validate', default=True, help='Validate code examples')
@click.option('--include-private/--no-private', default=False, help='Include private members')
def build(source_dir, docs_dir, output_dir, output_formats, validate, include_private):
    """Build complete documentation."""
    
    config = DocumentationConfig(
        source_dir=Path(source_dir),
        docs_dir=Path(docs_dir),
        build_dir=Path(output_dir),
        output_formats=list(output_formats),
        validate_code=validate,
        include_private=include_private
    )
    
    builder = DocumentationBuilder(config)
    asyncio.run(builder.build_documentation())


@cli.command()
@click.option('--watch/--no-watch', default=False, help='Watch for changes')
def serve(watch):
    """Serve documentation with auto-reload."""
    
    if watch:
        # Use watchdog to monitor changes
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class DocumentationHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith('.py') or event.src_path.endswith('.md'):
                    print(f"File changed: {event.src_path}")
                    # Rebuild documentation
                    config = DocumentationConfig()
                    builder = DocumentationBuilder(config)
                    asyncio.run(builder.build_documentation())
        
        observer = Observer()
        observer.schedule(DocumentationHandler(), 'src', recursive=True)
        observer.schedule(DocumentationHandler(), 'docs', recursive=True)
        observer.start()
        
        try:
            subprocess.run(['mkdocs', 'serve', '-f', 'docs/_build/mkdocs.yml'])
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        subprocess.run(['mkdocs', 'serve', '-f', 'docs/_build/mkdocs.yml'])


if __name__ == '__main__':
    cli()