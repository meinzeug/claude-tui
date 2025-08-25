"""
File Analyzer - File and project analysis utilities.

Provides file structure analysis, language detection, complexity calculation,
and code element extraction for AI context building.
"""

import ast
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class FileAnalyzer:
    """File and project analysis utilities."""
    
    def __init__(self):
        """Initialize file analyzer."""
        self._language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown',
            '.txt': 'text'
        }
    
    async def analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """
        Analyze project directory structure.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary representing project structure
        """
        if not project_path.exists() or not project_path.is_dir():
            return {}
        
        structure = {}
        
        try:
            for item in project_path.iterdir():
                if item.name.startswith('.') and item.name not in ['.env', '.gitignore']:
                    continue  # Skip hidden files except important ones
                
                if item.is_dir():
                    if item.name not in ['node_modules', '__pycache__', '.git', 'venv', '.venv']:
                        structure[item.name] = await self.analyze_project_structure(item)
                else:
                    structure[item.name] = {
                        'size': item.stat().st_size,
                        'language': self._detect_language(item),
                        'type': 'file'
                    }
            
            return structure
            
        except PermissionError:
            logger.warning(f"Permission denied accessing {project_path}")
            return {}
        except Exception as e:
            logger.error(f"Error analyzing project structure: {e}")
            return {}
    
    async def detect_language_and_framework(self, project_path: Path) -> Dict[str, Optional[str]]:
        """
        Detect primary language and framework of a project.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary with 'language' and 'framework' keys
        """
        result = {'language': None, 'framework': None}
        
        if not project_path.exists():
            return result
        
        # Count files by language
        language_counts = defaultdict(int)
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                language = self._detect_language(file_path)
                if language:
                    language_counts[language] += 1
        
        # Determine primary language
        if language_counts:
            result['language'] = max(language_counts, key=language_counts.get)
        
        # Detect framework based on files and language
        result['framework'] = await self._detect_framework(project_path, result['language'])
        
        return result
    
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Dictionary with file analysis results
        """
        if not file_path.exists() or not file_path.is_file():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Binary file
            return {
                'language': 'binary',
                'size': file_path.stat().st_size,
                'lines': 0
            }
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
        
        language = self._detect_language(file_path)
        
        return {
            'language': language,
            'size': len(content),
            'lines': len(content.split('\\n')),
            'dependencies': await self._extract_dependencies(content, language),
            'complexity': await self.calculate_complexity(file_path, content)
        }
    
    async def extract_code_elements(self, file_path: Path) -> Dict[str, List[str]]:
        """
        Extract code elements (functions, classes, etc.) from file.
        
        Args:
            file_path: Path to code file
            
        Returns:
            Dictionary with lists of code elements
        """
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return {}
        
        language = self._detect_language(file_path)
        
        if language == 'python':
            return await self._extract_python_elements(content)
        elif language in ['javascript', 'typescript']:
            return await self._extract_javascript_elements(content)
        else:
            return {}
    
    async def calculate_complexity(self, file_path: Path, content: Optional[str] = None) -> float:
        """
        Calculate code complexity score.
        
        Args:
            file_path: Path to code file
            content: File content (will read if None)
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                return 0.0
        
        language = self._detect_language(file_path)
        
        if language == 'python':
            return await self._calculate_python_complexity(content)
        elif language in ['javascript', 'typescript']:
            return await self._calculate_javascript_complexity(content)
        else:
            return await self._calculate_general_complexity(content)
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        return self._language_extensions.get(file_path.suffix.lower())
    
    async def _detect_framework(self, project_path: Path, language: Optional[str]) -> Optional[str]:
        """Detect framework based on project files and language."""
        framework_indicators = {
            'python': {
                'django': ['manage.py', 'settings.py', 'wsgi.py'],
                'flask': ['app.py', 'application.py'],
                'fastapi': ['main.py', 'app.py'],
                'pytest': ['pytest.ini', 'conftest.py'],
            },
            'javascript': {
                'react': ['package.json'],  # Will check content
                'vue': ['vue.config.js', 'nuxt.config.js'],
                'angular': ['angular.json', 'src/app/app.module.ts'],
                'express': ['package.json'],  # Will check content
                'next': ['next.config.js'],
            },
            'typescript': {
                'react': ['tsconfig.json'],
                'angular': ['angular.json', 'tsconfig.json'],
                'nest': ['nest-cli.json'],
            }
        }
        
        if not language or language not in framework_indicators:
            return None
        
        indicators = framework_indicators[language]
        
        for framework, files in indicators.items():
            for file_name in files:
                file_path = project_path / file_name
                if file_path.exists():
                    # For package.json, check content for specific frameworks
                    if file_name == 'package.json':
                        framework_name = await self._detect_js_framework_from_package_json(file_path)
                        if framework_name:
                            return framework_name
                    else:
                        return framework
        
        return None
    
    async def _detect_js_framework_from_package_json(self, package_json_path: Path) -> Optional[str]:
        """Detect JavaScript framework from package.json content."""
        try:
            import json
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            dependencies = {
                **package_data.get('dependencies', {}),
                **package_data.get('devDependencies', {})
            }
            
            if 'react' in dependencies:
                if 'next' in dependencies:
                    return 'next'
                return 'react'
            elif 'vue' in dependencies:
                return 'vue'
            elif 'express' in dependencies:
                return 'express'
            elif '@angular/core' in dependencies:
                return 'angular'
            
        except Exception:
            pass
        
        return None
    
    async def _extract_dependencies(self, content: str, language: Optional[str]) -> List[str]:
        """Extract dependencies from file content."""
        dependencies = []
        
        if language == 'python':
            # Extract Python imports
            import_patterns = [
                r'^import\\s+([\\w.]+)',
                r'^from\\s+([\\w.]+)\\s+import'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                dependencies.extend(matches)
        
        elif language in ['javascript', 'typescript']:
            # Extract JavaScript imports
            import_patterns = [
                r'import\\s+.*?from\\s+[\"\\']([^\"\\']+)[\"\\']',
                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dependencies.extend(matches)
        
        return dependencies
    
    async def _extract_python_elements(self, content: str) -> Dict[str, List[str]]:
        """Extract Python code elements."""
        elements = {
            'imports': [],
            'functions': [],
            'classes': [],
            'variables': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        elements['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        elements['imports'].append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    elements['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    elements['classes'].append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            elements['variables'].append(target.id)
        
        except SyntaxError:
            # Fallback to regex for invalid Python
            elements['functions'] = re.findall(r'^def\\s+(\\w+)', content, re.MULTILINE)
            elements['classes'] = re.findall(r'^class\\s+(\\w+)', content, re.MULTILINE)
        
        return elements
    
    async def _extract_javascript_elements(self, content: str) -> Dict[str, List[str]]:
        """Extract JavaScript code elements."""
        elements = {
            'imports': [],
            'functions': [],
            'classes': [],
            'variables': []
        }
        
        # Function patterns
        function_patterns = [
            r'function\\s+(\\w+)\\s*\\(',
            r'(\\w+)\\s*=\\s*function\\s*\\(',
            r'(\\w+)\\s*=\\s*\\([^)]*\\)\\s*=>',
            r'const\\s+(\\w+)\\s*=\\s*\\([^)]*\\)\\s*=>'
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            elements['functions'].extend(matches)
        
        # Class patterns
        class_matches = re.findall(r'class\\s+(\\w+)', content)
        elements['classes'].extend(class_matches)
        
        # Variable patterns
        var_patterns = [
            r'var\\s+(\\w+)',
            r'let\\s+(\\w+)',
            r'const\\s+(\\w+)'
        ]
        
        for pattern in var_patterns:
            matches = re.findall(pattern, content)
            elements['variables'].extend(matches)
        
        # Import patterns
        import_patterns = [
            r'import\\s+.*?from\\s+[\"\\']([^\"\\']+)[\"\\']',
            r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            elements['imports'].extend(matches)
        
        return elements
    
    async def _calculate_python_complexity(self, content: str) -> float:
        """Calculate Python code complexity."""
        try:
            tree = ast.parse(content)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, (ast.And, ast.Or)):
                    complexity += 1
            
            # Normalize to 0-1 range (rough estimate)
            lines = len(content.split('\\n'))
            normalized_complexity = min(complexity / max(lines, 1), 1.0)
            
            return normalized_complexity
            
        except SyntaxError:
            return 0.5  # Medium complexity for invalid syntax
    
    async def _calculate_javascript_complexity(self, content: str) -> float:
        """Calculate JavaScript code complexity."""
        complexity = 1  # Base complexity
        
        # Count control structures
        control_patterns = [
            r'\\bif\\b', r'\\belse\\b', r'\\bfor\\b', r'\\bwhile\\b',
            r'\\bswitch\\b', r'\\bcatch\\b', r'\\btry\\b', r'\\?.*:'
        ]
        
        for pattern in control_patterns:
            matches = re.findall(pattern, content)
            complexity += len(matches)
        
        # Normalize to 0-1 range
        lines = len(content.split('\\n'))
        normalized_complexity = min(complexity / max(lines, 1), 1.0)
        
        return normalized_complexity
    
    async def _calculate_general_complexity(self, content: str) -> float:
        """Calculate general code complexity for unknown languages."""
        lines = len(content.split('\\n'))
        chars = len(content)
        
        # Simple heuristic based on length and structure
        if lines < 10:
            return 0.1
        elif lines < 50:
            return 0.3
        elif lines < 200:
            return 0.5
        elif lines < 500:
            return 0.7
        else:
            return 0.9