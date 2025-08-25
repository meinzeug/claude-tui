"""
Context Builder - Intelligent context management for AI interactions.

Builds rich, contextual information for AI requests including project context,
code analysis, dependency mapping, and historical data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.project import Project
from claude_tiu.models.task import DevelopmentTask
from claude_tiu.utils.file_analyzer import FileAnalyzer
from claude_tiu.utils.dependency_mapper import DependencyMapper

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Project context information."""
    name: str
    path: str
    language: str
    framework: Optional[str]
    dependencies: List[str]
    file_structure: Dict[str, Any]
    recent_changes: List[Dict[str, Any]]
    metrics: Dict[str, float]


@dataclass
class CodeContext:
    """Code-specific context information."""
    file_path: str
    language: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    complexity_score: float
    dependencies: List[str]
    related_files: List[str]


class ContextBuilder:
    """
    Intelligent context management for AI interactions.
    
    The ContextBuilder analyzes projects and tasks to create rich contextual
    information that helps AI models generate more accurate and relevant responses.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the context builder.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        self.file_analyzer = FileAnalyzer()
        self.dependency_mapper = DependencyMapper()
        
        # Context caching
        self._project_contexts: Dict[str, ProjectContext] = {}
        self._code_contexts: Dict[str, CodeContext] = {}
        self._context_cache_ttl = 300  # 5 minutes
        
        logger.info("Context builder initialized")
    
    async def build_smart_context(
        self,
        prompt: str,
        context: Dict[str, Any],
        project: Optional[Project] = None
    ) -> Dict[str, Any]:
        """
        Build intelligent context for AI requests.
        
        Args:
            prompt: The AI prompt
            context: Base context information
            project: Associated project (optional)
            
        Returns:
            Dict containing enriched context
        """
        logger.debug("Building smart context for AI request")
        
        smart_context = context.copy()
        
        try:
            # Add project context if available
            if project:
                project_ctx = await self.build_project_context(project)
                smart_context['project'] = project_ctx
            
            # Analyze prompt for context hints
            prompt_analysis = await self._analyze_prompt(prompt)
            smart_context['prompt_analysis'] = prompt_analysis
            
            # Add relevant code context
            if 'file_path' in context:
                file_path = Path(context['file_path'])
                if file_path.exists():
                    code_ctx = await self.build_code_context(file_path, project)
                    smart_context['code'] = code_ctx
            
            # Add system context
            smart_context['system'] = await self._build_system_context()
            
            # Add user preferences
            smart_context['preferences'] = await self._build_preferences_context()
            
            logger.debug("Smart context built successfully")
            return smart_context
            
        except Exception as e:
            logger.error(f"Failed to build smart context: {e}")
            return context  # Return original context on failure
    
    async def build_project_context(self, project: Project) -> Dict[str, Any]:
        """
        Build comprehensive project context.
        
        Args:
            project: Project to analyze
            
        Returns:
            Dict containing project context
        """
        project_key = str(project.path)
        
        # Check cache first
        if project_key in self._project_contexts:
            cached_ctx = self._project_contexts[project_key]
            # Return cached if still valid (simplified TTL check)
            return cached_ctx.__dict__
        
        logger.debug(f"Analyzing project context: {project.name}")
        
        try:
            # Analyze project structure
            structure = await self.file_analyzer.analyze_project_structure(project.path)
            
            # Detect language and framework
            language_info = await self.file_analyzer.detect_language_and_framework(project.path)
            
            # Analyze dependencies
            dependencies = await self.dependency_mapper.extract_dependencies(project.path)
            
            # Get recent changes (simplified)
            recent_changes = await self._get_recent_changes(project.path)
            
            # Calculate metrics
            metrics = await self._calculate_project_metrics(project.path, structure)
            
            # Build context
            project_context = ProjectContext(
                name=project.name,
                path=str(project.path),
                language=language_info.get('language', 'unknown'),
                framework=language_info.get('framework'),
                dependencies=dependencies,
                file_structure=structure,
                recent_changes=recent_changes,
                metrics=metrics
            )
            
            # Cache the context
            self._project_contexts[project_key] = project_context
            
            logger.debug(f"Project context built for {project.name}")
            return project_context.__dict__
            
        except Exception as e:
            logger.error(f"Failed to build project context: {e}")
            return {
                'name': project.name,
                'path': str(project.path),
                'error': str(e)
            }
    
    async def build_code_context(
        self,
        file_path: Path,
        project: Optional[Project] = None
    ) -> Dict[str, Any]:
        """
        Build code-specific context.
        
        Args:
            file_path: Path to code file
            project: Associated project (optional)
            
        Returns:
            Dict containing code context
        """
        file_key = str(file_path)
        
        # Check cache
        if file_key in self._code_contexts:
            cached_ctx = self._code_contexts[file_key]
            return cached_ctx.__dict__
        
        logger.debug(f"Analyzing code context: {file_path.name}")
        
        try:
            # Analyze file content
            file_analysis = await self.file_analyzer.analyze_file(file_path)
            
            # Extract code elements
            code_elements = await self.file_analyzer.extract_code_elements(file_path)
            
            # Calculate complexity
            complexity = await self.file_analyzer.calculate_complexity(file_path)
            
            # Find related files
            related_files = []
            if project:
                related_files = await self.dependency_mapper.find_related_files(
                    file_path, project.path
                )
            
            # Build code context
            code_context = CodeContext(
                file_path=str(file_path),
                language=file_analysis.get('language', 'unknown'),
                imports=code_elements.get('imports', []),
                functions=code_elements.get('functions', []),
                classes=code_elements.get('classes', []),
                complexity_score=complexity,
                dependencies=file_analysis.get('dependencies', []),
                related_files=[str(f) for f in related_files]
            )
            
            # Cache the context
            self._code_contexts[file_key] = code_context
            
            logger.debug(f"Code context built for {file_path.name}")
            return code_context.__dict__
            
        except Exception as e:
            logger.error(f"Failed to build code context for {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e)
            }
    
    async def build_task_context(
        self,
        task: DevelopmentTask,
        project: Project
    ) -> Dict[str, Any]:
        """
        Build task-specific context.
        
        Args:
            task: Development task
            project: Associated project
            
        Returns:
            Dict containing task context
        """
        logger.debug(f"Building task context: {task.name}")
        
        try:
            # Base task context
            task_context = {
                'task': {
                    'name': task.name,
                    'description': task.description,
                    'type': task.task_type.value if task.task_type else 'unknown',
                    'priority': task.priority.value if task.priority else 'medium'
                }
            }
            
            # Add project context
            project_ctx = await self.build_project_context(project)
            task_context['project'] = project_ctx
            
            # Add task-specific code context if file path specified
            if hasattr(task, 'file_path') and task.file_path:
                code_ctx = await self.build_code_context(task.file_path, project)
                task_context['code'] = code_ctx
            
            # Add dependencies context
            if hasattr(task, 'dependencies') and task.dependencies:
                dep_context = await self._build_dependencies_context(
                    task.dependencies, project
                )
                task_context['dependencies'] = dep_context
            
            return task_context
            
        except Exception as e:
            logger.error(f"Failed to build task context: {e}")
            return {'error': str(e)}
    
    async def invalidate_cache(self, project: Optional[Project] = None) -> None:
        """
        Invalidate context cache.
        
        Args:
            project: Specific project to invalidate (all if None)
        """
        if project:
            project_key = str(project.path)
            self._project_contexts.pop(project_key, None)
            
            # Invalidate related code contexts
            project_path_str = str(project.path)
            to_remove = [
                key for key in self._code_contexts.keys()
                if key.startswith(project_path_str)
            ]
            for key in to_remove:
                del self._code_contexts[key]
        else:
            self._project_contexts.clear()
            self._code_contexts.clear()
        
        logger.info("Context cache invalidated")
    
    # Private helper methods
    
    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt for context hints.
        """
        analysis = {
            'length': len(prompt),
            'words': len(prompt.split()),
            'contains_code': '```' in prompt or 'def ' in prompt or 'function' in prompt,
            'contains_file_ref': any(ext in prompt.lower() for ext in ['.py', '.js', '.ts', '.json']),
            'task_type': 'unknown'
        }
        
        # Simple task type detection
        prompt_lower = prompt.lower()
        if 'create' in prompt_lower or 'generate' in prompt_lower:
            analysis['task_type'] = 'generation'
        elif 'fix' in prompt_lower or 'debug' in prompt_lower:
            analysis['task_type'] = 'debugging'
        elif 'refactor' in prompt_lower or 'improve' in prompt_lower:
            analysis['task_type'] = 'refactoring'
        elif 'test' in prompt_lower:
            analysis['task_type'] = 'testing'
        elif 'review' in prompt_lower:
            analysis['task_type'] = 'review'
        
        return analysis
    
    async def _build_system_context(self) -> Dict[str, Any]:
        """
        Build system context information.
        """
        import platform
        import psutil
        
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _build_preferences_context(self) -> Dict[str, Any]:
        """
        Build user preferences context.
        """
        try:
            preferences = self.config_manager.get_ui_preferences()
            return {
                'theme': preferences.theme,
                'log_level': preferences.log_level,
                'show_progress_details': preferences.show_progress_details
            }
        except Exception:
            return {'error': 'Failed to load preferences'}
    
    async def _get_recent_changes(self, project_path: Path) -> List[Dict[str, Any]]:
        """
        Get recent project changes (simplified implementation).
        """
        try:
            # This would integrate with git or file system monitoring
            # For now, return empty list
            return []
        except Exception:
            return []
    
    async def _calculate_project_metrics(
        self,
        project_path: Path,
        structure: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate basic project metrics.
        """
        try:
            total_files = 0
            total_size = 0
            
            def count_files(node):
                nonlocal total_files, total_size
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                            total_files += 1
                            # Simplified size calculation
                            total_size += len(str(value)) if isinstance(value, str) else 100
                        elif isinstance(value, dict):
                            count_files(value)
            
            count_files(structure)
            
            return {
                'file_count': total_files,
                'estimated_size_kb': total_size / 1024,
                'complexity_score': min(total_files / 50, 1.0)  # Simplified
            }
            
        except Exception:
            return {'error': 'Failed to calculate metrics'}
    
    async def _build_dependencies_context(
        self,
        dependencies: List[str],
        project: Project
    ) -> Dict[str, Any]:
        """
        Build context for task dependencies.
        """
        try:
            dep_context = {
                'count': len(dependencies),
                'dependencies': []
            }
            
            for dep_id in dependencies[:5]:  # Limit to first 5
                # This would look up actual dependency information
                dep_context['dependencies'].append({
                    'id': dep_id,
                    'status': 'unknown'
                })
            
            return dep_context
            
        except Exception as e:
            return {'error': str(e)}