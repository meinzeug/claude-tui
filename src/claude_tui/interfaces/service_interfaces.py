"""Service Interface Definitions."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize configuration system."""
        self._config_data = {}
        self._initialized = True
        logger.info("Configuration system initialized")
    
    @abstractmethod
    async def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        import json
        
        if not config_path:
            config_path = Path("config.json")
            
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._config_data = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                self._config_data = self._get_default_config()
                logger.info("Using default configuration")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config_data = self._get_default_config()
            
        return self._config_data
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'debug': False,
            'log_level': 'INFO',
            'api': {
                'timeout': 30,
                'retries': 3
            },
            'ui': {
                'theme': 'default',
                'auto_save': True
            }
        }
    
    @abstractmethod
    async def save_config(self, config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        import json
        
        if not config_path:
            config_path = Path("config.json")
            
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self._config_data = config
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if not hasattr(self, '_config_data'):
            self._config_data = {}
            
        # Support nested keys with dot notation
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        if not hasattr(self, '_config_data'):
            self._config_data = {}
            
        # Support nested keys with dot notation
        keys = key.split('.')
        config = self._config_data
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the final value
        config[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")
    
    @abstractmethod
    async def reload(self) -> None:
        """Reload configuration from file."""
        await self.load_config()
        logger.info("Configuration reloaded")


class AIInterface(ABC):
    """Interface for AI service interactions."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize AI service."""
        self._client = None
        self._initialized = True
        logger.info("AI service initialized")
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "claude-3-sonnet",
        **kwargs
    ) -> Dict[str, Any]:
        """Get chat completion from AI service."""
        if not hasattr(self, '_client') or not self._client:
            raise RuntimeError("AI client not initialized")
            
        try:
            # Simulate API call structure
            response = {
                'id': f'msg_{hash(str(messages))}_completion',
                'content': [
                    {
                        'type': 'text',
                        'text': 'This is a placeholder AI response. The actual implementation would call the Claude API.'
                    }
                ],
                'model': model,
                'role': 'assistant',
                'stop_reason': 'end_turn',
                'usage': {
                    'input_tokens': sum(len(msg.get('content', '').split()) for msg in messages),
                    'output_tokens': 20
                }
            }
            
            logger.debug(f"Chat completion with {model}, {len(messages)} messages")
            return response
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    @abstractmethod
    async def stream_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "claude-3-sonnet",
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream chat completion from AI service."""
        import asyncio
        
        if not hasattr(self, '_client') or not self._client:
            raise RuntimeError("AI client not initialized")
            
        # Simulate streaming response
        response_text = "This is a simulated streaming AI response. "
        response_text += "Each word would normally come from the Claude API stream."
        
        words = response_text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)  # Simulate streaming delay
            
        logger.debug(f"Streamed completion with {model}")
    
    @abstractmethod
    async def code_completion(
        self, 
        code: str, 
        language: str = "python",
        **kwargs
    ) -> List[str]:
        """Get code completion suggestions."""
        # Simulate code completion based on common patterns
        suggestions = []
        
        if language == "python":
            if "def " in code:
                suggestions.extend(["return None", "pass", "raise NotImplementedError"])
            elif "class " in code:
                suggestions.extend(["def __init__(self):", "pass"])
            elif "if " in code:
                suggestions.extend(["pass", "return True", "break"])
            else:
                suggestions.extend(["print()", "return", "import "])
        elif language == "javascript":
            suggestions.extend(["console.log()", "return", "const ", "function"])
        else:
            suggestions.append("// Implementation needed")
            
        logger.debug(f"Code completion for {language}: {len(suggestions)} suggestions")
        return suggestions
    
    @abstractmethod
    async def code_review(
        self, 
        code: str, 
        language: str = "python",
        **kwargs
    ) -> Dict[str, Any]:
        """Get code review and suggestions."""
        # Simulate basic code analysis
        issues = []
        suggestions = []
        
        # Basic static analysis
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'pass' in line and len(line.strip()) == 4:
                issues.append({
                    'line': i,
                    'type': 'placeholder',
                    'severity': 'medium',
                    'message': 'Empty pass statement - needs implementation'
                })
            if 'TODO' in line:
                issues.append({
                    'line': i,
                    'type': 'todo',
                    'severity': 'low', 
                    'message': 'Implementation comment found'
                })
        
        # Generate suggestions
        if not issues:
            suggestions.append("Code looks good! Consider adding more comments.")
        else:
            suggestions.append(f"Found {len(issues)} issues to address.")
            
        review_result = {
            'overall_score': max(8 - len(issues), 3),
            'issues': issues,
            'suggestions': suggestions,
            'language': language,
            'lines_reviewed': len(lines)
        }
        
        logger.debug(f"Code review completed: {len(issues)} issues found")
        return review_result
    
    @abstractmethod
    async def explain_code(
        self, 
        code: str, 
        language: str = "python",
        **kwargs
    ) -> str:
        """Get code explanation."""
        # Generate a basic code explanation
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        explanation = f"This {language} code snippet contains {len(non_empty_lines)} lines of code. "
        
        # Basic pattern recognition
        if 'def ' in code:
            func_count = code.count('def ')
            explanation += f"It defines {func_count} function{'s' if func_count > 1 else ''}. "
            
        if 'class ' in code:
            class_count = code.count('class ')
            explanation += f"It contains {class_count} class definition{'s' if class_count > 1 else ''}. "
            
        if 'import ' in code:
            explanation += "It includes import statements for external dependencies. "
            
        if 'if ' in code:
            explanation += "It contains conditional logic with if statements. "
            
        if 'for ' in code or 'while ' in code:
            explanation += "It includes loop structures for iteration. "
            
        explanation += "\n\nNote: This is a basic analysis. A full AI implementation would provide more detailed explanations."
        
        logger.debug(f"Code explanation generated for {len(lines)} lines")
        return explanation


class ProjectInterface(ABC):
    """Interface for project management."""
    
    @abstractmethod
    async def initialize(self, project_path: Path) -> None:
        """Initialize project at path."""
        self._project_path = project_path
        self._project_files = []
        self._project_metadata = {}
        
        if not project_path.exists():
            project_path.mkdir(parents=True, exist_ok=True)
            
        # Create basic project structure
        (project_path / "src").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "docs").mkdir(exist_ok=True)
        
        logger.info(f"Initialized project at {project_path}")
    
    @abstractmethod
    async def load_project(self, project_path: Path) -> Dict[str, Any]:
        """Load project from path."""
        import json
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
            
        self._project_path = project_path
        
        # Load project metadata if exists
        metadata_file = project_path / "project.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._project_metadata = json.load(f)
        else:
            self._project_metadata = {
                'name': project_path.name,
                'version': '0.1.0',
                'created': str(datetime.now()),
                'type': 'unknown'
            }
            
        # Scan for project files
        self._project_files = list(project_path.rglob('*'))
        self._project_files = [f for f in self._project_files if f.is_file()]
        
        logger.info(f"Loaded project {self._project_metadata.get('name', 'Unknown')} with {len(self._project_files)} files")
        return self._project_metadata
    
    @abstractmethod
    async def save_project(self) -> None:
        """Save current project state."""
        import json
        from datetime import datetime
        
        if not hasattr(self, '_project_path'):
            raise RuntimeError("No project loaded")
            
        self._project_metadata['last_saved'] = str(datetime.now())
        
        metadata_file = self._project_path / "project.json"
        with open(metadata_file, 'w') as f:
            json.dump(self._project_metadata, f, indent=2)
            
        logger.info(f"Saved project state to {metadata_file}")
    
    @abstractmethod
    def get_project_files(self) -> List[Path]:
        """Get list of project files."""
        if hasattr(self, '_project_files'):
            return self._project_files.copy()
        return []
    
    @abstractmethod
    async def add_file(self, file_path: Path) -> None:
        """Add file to project."""
        if not hasattr(self, '_project_files'):
            self._project_files = []
            
        if file_path not in self._project_files:
            self._project_files.append(file_path)
            logger.debug(f"Added file to project: {file_path}")
    
    @abstractmethod
    async def remove_file(self, file_path: Path) -> None:
        """Remove file from project."""
        if hasattr(self, '_project_files') and file_path in self._project_files:
            self._project_files.remove(file_path)
            # Also remove from filesystem if exists
            if file_path.exists():
                file_path.unlink()
            logger.debug(f"Removed file from project: {file_path}")
    
    @abstractmethod
    async def rename_file(self, old_path: Path, new_path: Path) -> None:
        """Rename file in project."""
        if hasattr(self, '_project_files'):
            try:
                index = self._project_files.index(old_path)
                self._project_files[index] = new_path
                
                # Rename in filesystem
                if old_path.exists():
                    old_path.rename(new_path)
                    
                logger.debug(f"Renamed {old_path} to {new_path}")
            except ValueError:
                logger.warning(f"File {old_path} not found in project")
    
    @abstractmethod
    def get_project_info(self) -> Dict[str, Any]:
        """Get project metadata."""
        info = getattr(self, '_project_metadata', {}).copy()
        info['file_count'] = len(getattr(self, '_project_files', []))
        info['path'] = str(getattr(self, '_project_path', ''))
        return info
    
    @abstractmethod
    async def run_command(self, command: str, cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run command in project context."""
        import subprocess
        import asyncio
        from datetime import datetime
        
        if not cwd:
            cwd = getattr(self, '_project_path', Path.cwd())
            
        try:
            start_time = datetime.now()
            
            # Run command asynchronously
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            end_time = datetime.now()
            
            result = {
                'command': command,
                'return_code': proc.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'execution_time': (end_time - start_time).total_seconds(),
                'success': proc.returncode == 0
            }
            
            logger.debug(f"Executed command: {command} (exit code: {proc.returncode})")
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'command': command,
                'return_code': -1,
                'error': str(e),
                'success': False
            }


class ValidationInterface(ABC):
    """Interface for code validation services."""
    
    @abstractmethod
    async def validate_code(
        self, 
        code: str, 
        language: str = "python",
        file_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Validate code and return issues."""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for common issues
            if line_stripped == 'pass' and line_num > 1:
                issues.append({
                    'line': line_num,
                    'column': line.index('pass') + 1,
                    'type': 'placeholder',
                    'severity': 'medium',
                    'message': 'Empty pass statement needs implementation',
                    'file': str(file_path) if file_path else None
                })
                
            if 'TODO' in line:
                issues.append({
                    'line': line_num,
                    'column': line.index('TODO') + 1,
                    'type': 'todo',
                    'severity': 'low',
                    'message': 'Implementation comment found',
                    'file': str(file_path) if file_path else None
                })
                
            if 'NotImplementedError' in line:
                issues.append({
                    'line': line_num,
                    'column': line.index('NotImplementedError') + 1,
                    'type': 'not_implemented',
                    'severity': 'high',
                    'message': 'NotImplementedError - functionality missing',
                    'file': str(file_path) if file_path else None
                })
        
        logger.debug(f"Code validation found {len(issues)} issues")
        return issues
    
    @abstractmethod
    async def validate_project(self, project_path: Path) -> Dict[str, Any]:
        """Validate entire project."""
        if not project_path.exists():
            return {
                'valid': False,
                'error': f'Project path does not exist: {project_path}'
            }
            
        all_issues = []
        files_validated = 0
        
        # Find and validate all code files
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in code_extensions:
                try:
                    code = file_path.read_text(encoding='utf-8')
                    language = self._get_language_from_extension(file_path.suffix)
                    file_issues = await self.validate_code(code, language, file_path)
                    all_issues.extend(file_issues)
                    files_validated += 1
                except Exception as e:
                    logger.warning(f"Failed to validate {file_path}: {e}")
                    
        # Categorize issues by severity
        high_issues = [i for i in all_issues if i.get('severity') == 'high']
        medium_issues = [i for i in all_issues if i.get('severity') == 'medium']
        low_issues = [i for i in all_issues if i.get('severity') == 'low']
        
        return {
            'valid': len(high_issues) == 0,
            'files_validated': files_validated,
            'total_issues': len(all_issues),
            'high_severity_issues': len(high_issues),
            'medium_severity_issues': len(medium_issues),
            'low_severity_issues': len(low_issues),
            'issues': all_issues,
            'project_path': str(project_path)
        }
        
    def _get_language_from_extension(self, extension: str) -> str:
        """Map file extension to language name."""
        mapping = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return mapping.get(extension, 'unknown')
    
    @abstractmethod
    async def fix_issues(
        self, 
        issues: List[Dict[str, Any]], 
        code: str,
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """Fix validation issues."""
        if not auto_fix:
            return {
                'fixed': False,
                'message': 'Auto-fix disabled, no changes made',
                'code': code,
                'issues_fixed': 0
            }
            
        fixed_code = code
        lines = fixed_code.split('\n')
        issues_fixed = 0
        
        # Sort issues by line number (descending) to avoid line number shifts
        sorted_issues = sorted(issues, key=lambda x: x.get('line', 0), reverse=True)
        
        for issue in sorted_issues:
            line_num = issue.get('line', 0) - 1  # Convert to 0-based
            if 0 <= line_num < len(lines):
                issue_type = issue.get('type')
                
                if issue_type == 'placeholder' and 'pass' in lines[line_num]:
                    # Replace pass with a proper implementation stub
                    indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                    lines[line_num] = ' ' * indent + 'raise NotImplementedError("Method implementation required")'
                    issues_fixed += 1
                    
                elif issue_type == 'not_implemented':
                    # Add implementation comment above NotImplementedError
                    indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                    todo_line = ' ' * indent + '# Implementation: Add functionality here'
                    lines.insert(line_num, todo_line)
                    issues_fixed += 1
        
        fixed_code = '\n'.join(lines)
        
        return {
            'fixed': issues_fixed > 0,
            'message': f'Fixed {issues_fixed} issues automatically',
            'code': fixed_code,
            'issues_fixed': issues_fixed,
            'original_issues': len(issues)
        }
    
    @abstractmethod
    async def check_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Check project dependencies."""
        result = {
            'dependencies_found': [],
            'missing_dependencies': [],
            'dependency_files': [],
            'status': 'unknown'
        }
        
        # Check common dependency files
        dep_files = {
            'requirements.txt': 'python',
            'package.json': 'node',
            'Cargo.toml': 'rust',
            'go.mod': 'go',
            'pom.xml': 'java'
        }
        
        for dep_file, lang in dep_files.items():
            file_path = project_path / dep_file
            if file_path.exists():
                result['dependency_files'].append({
                    'file': dep_file,
                    'language': lang,
                    'exists': True
                })
                
                # Parse dependencies (basic implementation)
                try:
                    content = file_path.read_text()
                    if dep_file == 'requirements.txt':
                        deps = [line.strip() for line in content.split('\n') 
                               if line.strip() and not line.startswith('#')]
                        result['dependencies_found'].extend(deps)
                    elif dep_file == 'package.json':
                        import json
                        data = json.loads(content)
                        deps = list(data.get('dependencies', {}).keys())
                        result['dependencies_found'].extend(deps)
                except Exception as e:
                    logger.warning(f"Failed to parse {dep_file}: {e}")
                    
        result['status'] = 'found' if result['dependency_files'] else 'none_found'
        result['total_dependencies'] = len(result['dependencies_found'])
        
        logger.debug(f"Dependency check: {len(result['dependency_files'])} files, {len(result['dependencies_found'])} deps")
        return result
    
    @abstractmethod
    async def detect_placeholders(
        self, 
        code: str, 
        language: str = "python"
    ) -> List[Dict[str, Any]]:
        """Detect placeholder implementations."""
        placeholders = []
        lines = code.split('\n')
        
        placeholder_patterns = {
            'python': [
                ('pass', 'empty_pass', 'Empty pass statement'),
                ('TODO', 'todo_comment', 'Implementation comment'),
                ('FIXME', 'fixme_comment', 'Fix required comment'), 
                ('NotImplementedError', 'not_implemented', 'NotImplementedError raised'),
                ('raise NotImplementedError', 'not_implemented_raise', 'Explicit NotImplementedError')
            ],
            'javascript': [
                ('// TODO', 'todo_comment', 'Implementation comment'),
                ('throw new Error', 'error_throw', 'Error thrown'),
                ('console.log', 'debug_log', 'Debug console.log')
            ]
        }
        
        patterns = placeholder_patterns.get(language, placeholder_patterns['python'])
        
        for line_num, line in enumerate(lines, 1):
            for pattern, ptype, description in patterns:
                if pattern in line:
                    placeholders.append({
                        'line': line_num,
                        'column': line.index(pattern) + 1,
                        'type': ptype,
                        'pattern': pattern,
                        'description': description,
                        'context': line.strip(),
                        'language': language
                    })
        
        logger.debug(f"Detected {len(placeholders)} placeholders in {language} code")
        return placeholders