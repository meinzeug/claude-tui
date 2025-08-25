"""
Dependency Mapper - Project dependency analysis and mapping utilities.

Provides dependency extraction, relationship mapping, and file relationship
analysis for AI context building.
"""

import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DependencyMapper:
    """Project dependency analysis and mapping utilities."""
    
    def __init__(self):
        """Initialize dependency mapper."""
        self._dependency_files = {
            'python': ['requirements.txt', 'pyproject.toml', 'Pipfile', 'setup.py'],
            'javascript': ['package.json', 'package-lock.json', 'yarn.lock'],
            'java': ['pom.xml', 'build.gradle'],
            'csharp': ['*.csproj', 'packages.config'],
            'go': ['go.mod', 'go.sum'],
            'rust': ['Cargo.toml', 'Cargo.lock']
        }
    
    async def extract_dependencies(self, project_path: Path) -> List[str]:
        """
        Extract project dependencies from configuration files.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            List of dependency names
        """
        dependencies = []
        
        for language, dep_files in self._dependency_files.items():
            for dep_file_pattern in dep_files:
                if '*' in dep_file_pattern:
                    # Handle glob patterns
                    matching_files = list(project_path.glob(dep_file_pattern))
                else:
                    matching_files = [project_path / dep_file_pattern]
                
                for dep_file in matching_files:
                    if dep_file.exists():
                        file_deps = await self._extract_from_file(dep_file, language)
                        dependencies.extend(file_deps)
        
        return list(set(dependencies))  # Remove duplicates
    
    async def find_related_files(
        self,
        target_file: Path,
        project_path: Path
    ) -> List[Path]:
        """
        Find files related to the target file through imports/dependencies.
        
        Args:
            target_file: File to find relationships for
            project_path: Project root directory
            
        Returns:
            List of related file paths
        """
        related_files = []
        
        try:
            # Read target file content
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract imports/requires from target file
            imports = await self._extract_file_imports(content, target_file)
            
            # Find corresponding files in project
            for import_path in imports:
                related_file = await self._resolve_import_to_file(
                    import_path, target_file, project_path
                )
                if related_file and related_file.exists():
                    related_files.append(related_file)
            
            # Also find files that import the target file
            reverse_deps = await self._find_reverse_dependencies(
                target_file, project_path
            )
            related_files.extend(reverse_deps)
            
        except Exception as e:
            logger.error(f"Error finding related files for {target_file}: {e}")
        
        return list(set(related_files))  # Remove duplicates
    
    async def build_dependency_graph(self, project_path: Path) -> Dict[str, List[str]]:
        """
        Build a dependency graph for the project.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary mapping files to their dependencies
        """
        dependency_graph = defaultdict(list)
        
        # Find all relevant source files
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs', '.go', '.rs'}
        
        for file_path in project_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in source_extensions and
                not any(part.startswith('.') for part in file_path.parts)):
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    imports = await self._extract_file_imports(content, file_path)
                    
                    # Convert imports to file paths
                    file_dependencies = []
                    for import_path in imports:
                        dep_file = await self._resolve_import_to_file(
                            import_path, file_path, project_path
                        )
                        if dep_file:
                            file_dependencies.append(str(dep_file.relative_to(project_path)))
                    
                    if file_dependencies:
                        dependency_graph[str(file_path.relative_to(project_path))] = file_dependencies
                
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        return dict(dependency_graph)
    
    async def _extract_from_file(self, dep_file: Path, language: str) -> List[str]:
        """Extract dependencies from a specific dependency file."""
        dependencies = []
        
        try:
            if dep_file.name == 'requirements.txt':
                dependencies = await self._extract_from_requirements_txt(dep_file)
            elif dep_file.name == 'package.json':
                dependencies = await self._extract_from_package_json(dep_file)
            elif dep_file.name == 'pyproject.toml':
                dependencies = await self._extract_from_pyproject_toml(dep_file)
            elif dep_file.name == 'Cargo.toml':
                dependencies = await self._extract_from_cargo_toml(dep_file)
            elif dep_file.name == 'go.mod':
                dependencies = await self._extract_from_go_mod(dep_file)
            # Add more parsers as needed
        
        except Exception as e:
            logger.error(f"Error extracting dependencies from {dep_file}: {e}")
        
        return dependencies
    
    async def _extract_from_requirements_txt(self, file_path: Path) -> List[str]:
        """Extract dependencies from requirements.txt."""
        dependencies = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifiers)
                    package = re.split(r'[>=<!=]', line)[0].strip()
                    if package:
                        dependencies.append(package)
        
        return dependencies
    
    async def _extract_from_package_json(self, file_path: Path) -> List[str]:
        """Extract dependencies from package.json."""
        dependencies = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        # Extract from dependencies and devDependencies
        for dep_section in ['dependencies', 'devDependencies', 'peerDependencies']:
            if dep_section in package_data:
                dependencies.extend(package_data[dep_section].keys())
        
        return dependencies
    
    async def _extract_from_pyproject_toml(self, file_path: Path) -> List[str]:
        """Extract dependencies from pyproject.toml."""
        dependencies = []
        
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                pyproject_data = toml.load(f)
            
            # Extract from tool.poetry.dependencies or project.dependencies
            if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                poetry_deps = pyproject_data['tool']['poetry'].get('dependencies', {})
                dependencies.extend([dep for dep in poetry_deps.keys() if dep != 'python'])
            
            if 'project' in pyproject_data:
                project_deps = pyproject_data['project'].get('dependencies', [])
                for dep in project_deps:
                    # Extract package name from requirement string
                    package = re.split(r'[>=<!=]', dep)[0].strip()
                    dependencies.append(package)
        
        except ImportError:
            logger.warning("toml library not available for pyproject.toml parsing")
        
        return dependencies
    
    async def _extract_from_cargo_toml(self, file_path: Path) -> List[str]:
        """Extract dependencies from Cargo.toml."""
        dependencies = []
        
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                cargo_data = toml.load(f)
            
            if 'dependencies' in cargo_data:
                dependencies.extend(cargo_data['dependencies'].keys())
            
            if 'dev-dependencies' in cargo_data:
                dependencies.extend(cargo_data['dev-dependencies'].keys())
        
        except ImportError:
            logger.warning("toml library not available for Cargo.toml parsing")
        
        return dependencies
    
    async def _extract_from_go_mod(self, file_path: Path) -> List[str]:
        """Extract dependencies from go.mod."""
        dependencies = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract require statements
        require_matches = re.findall(r'require\\s+([^\\s]+)', content)
        dependencies.extend(require_matches)
        
        return dependencies
    
    async def _extract_file_imports(self, content: str, file_path: Path) -> List[str]:
        """Extract imports from file content based on language."""
        imports = []
        language = self._detect_language(file_path)
        
        if language == 'python':
            import_patterns = [
                r'^import\\s+([\\w.]+)',
                r'^from\\s+([\\w.]+)\\s+import'
            ]
        elif language in ['javascript', 'typescript']:
            import_patterns = [
                r'import\\s+.*?from\\s+[\"\\']([^\"\\']+)[\"\\']',
                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'
            ]
        elif language == 'java':
            import_patterns = [r'^import\\s+([\\w.]+);']
        elif language == 'csharp':
            import_patterns = [r'^using\\s+([\\w.]+);']
        elif language == 'go':
            import_patterns = [r'import\\s+\"([^\"]+)\"']
        else:
            return imports
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
        
        return imports
    
    async def _resolve_import_to_file(
        self,
        import_path: str,
        source_file: Path,
        project_path: Path
    ) -> Optional[Path]:
        """Resolve an import path to an actual file."""
        language = self._detect_language(source_file)
        
        if language == 'python':
            return await self._resolve_python_import(import_path, source_file, project_path)
        elif language in ['javascript', 'typescript']:
            return await self._resolve_js_import(import_path, source_file, project_path)
        
        return None
    
    async def _resolve_python_import(
        self,
        import_path: str,
        source_file: Path,
        project_path: Path
    ) -> Optional[Path]:
        """Resolve Python import to file path."""
        # Handle relative imports
        if import_path.startswith('.'):
            # Relative import - resolve based on source file location
            source_dir = source_file.parent
            parts = import_path.lstrip('.').split('.')
            
            # Go up directories for each leading dot
            dots = len(import_path) - len(import_path.lstrip('.'))
            target_dir = source_dir
            for _ in range(dots - 1):
                target_dir = target_dir.parent
            
            # Navigate to module
            for part in parts:
                if part:
                    target_dir = target_dir / part
            
            # Try .py file or __init__.py in directory
            if (target_dir.with_suffix('.py')).exists():
                return target_dir.with_suffix('.py')
            elif (target_dir / '__init__.py').exists():
                return target_dir / '__init__.py'
        
        else:
            # Absolute import - search in project
            parts = import_path.split('.')
            potential_path = project_path
            
            for part in parts:
                potential_path = potential_path / part
            
            # Try .py file or __init__.py in directory
            if potential_path.with_suffix('.py').exists():
                return potential_path.with_suffix('.py')
            elif (potential_path / '__init__.py').exists():
                return potential_path / '__init__.py'
        
        return None
    
    async def _resolve_js_import(
        self,
        import_path: str,
        source_file: Path,
        project_path: Path
    ) -> Optional[Path]:
        """Resolve JavaScript/TypeScript import to file path."""
        # Handle relative imports
        if import_path.startswith('.'):
            source_dir = source_file.parent
            resolved_path = (source_dir / import_path).resolve()
            
            # Try different extensions
            extensions = ['.js', '.ts', '.jsx', '.tsx', '.json']
            
            for ext in extensions:
                if resolved_path.with_suffix(ext).exists():
                    return resolved_path.with_suffix(ext)
            
            # Try index files
            for ext in extensions:
                index_file = resolved_path / f'index{ext}'
                if index_file.exists():
                    return index_file
        
        else:
            # Could be a node_modules import or absolute path
            # For now, we don't resolve these
            pass
        
        return None
    
    async def _find_reverse_dependencies(
        self,
        target_file: Path,
        project_path: Path
    ) -> List[Path]:
        """Find files that depend on the target file."""
        reverse_deps = []
        
        # Get relative path for matching
        try:
            target_rel_path = target_file.relative_to(project_path)
        except ValueError:
            return reverse_deps
        
        # Search through project files
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        for file_path in project_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in source_extensions and
                file_path != target_file):
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if this file imports the target file
                    imports = await self._extract_file_imports(content, file_path)
                    
                    for import_path in imports:
                        resolved_file = await self._resolve_import_to_file(
                            import_path, file_path, project_path
                        )
                        if resolved_file == target_file:
                            reverse_deps.append(file_path)
                            break
                
                except Exception:
                    continue
        
        return reverse_deps
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust'
        }
        return extension_map.get(file_path.suffix.lower())