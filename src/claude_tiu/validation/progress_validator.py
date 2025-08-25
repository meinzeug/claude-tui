"""
Progress Validator - Advanced validation system for AI-generated content.

Implements comprehensive anti-hallucination validation including:
- Placeholder detection
- Semantic analysis  
- Execution testing
- Cross-validation
- Auto-fixing capabilities
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.models.project import Project
from claude_tiu.models.task import DevelopmentTask, TaskResult
from claude_tiu.validation.placeholder_detector import PlaceholderDetector
from claude_tiu.validation.semantic_analyzer import SemanticAnalyzer
from claude_tiu.validation.execution_tester import ExecutionTester
from claude_tiu.validation.auto_completion_engine import AutoCompletionEngine

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    id: str
    description: str
    severity: ValidationSeverity
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    issue_type: str = "unknown"
    auto_fixable: bool = False
    suggested_fix: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    overall_score: float
    authenticity_score: float
    completeness_score: float
    quality_score: float
    issues: List[ValidationIssue]
    summary: str
    execution_time: float
    validated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'is_valid': self.is_valid,
            'overall_score': self.overall_score,
            'authenticity_score': self.authenticity_score,
            'completeness_score': self.completeness_score,
            'quality_score': self.quality_score,
            'issues_count': len(self.issues),
            'critical_issues': len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]),
            'high_issues': len([i for i in self.issues if i.severity == ValidationSeverity.HIGH]),
            'summary': self.summary,
            'execution_time': self.execution_time,
            'validated_at': self.validated_at.isoformat()
        }


class ProgressValidator:
    """
    Advanced validation system for AI-generated content.
    
    Implements multi-layered validation to detect hallucinations, incomplete
    implementations, and quality issues in AI-generated code and content.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the progress validator.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        
        # Initialize validation components
        self.placeholder_detector = PlaceholderDetector(config_manager)
        self.semantic_analyzer = SemanticAnalyzer(config_manager)
        self.execution_tester = ExecutionTester(config_manager)
        self.auto_completion_engine = AutoCompletionEngine(config_manager)
        
        # Validation configuration
        self._validation_enabled = True
        self._auto_fix_enabled = True
        self._quality_threshold = 0.7
        self._authenticity_threshold = 0.8
        
        # Runtime state
        self._validation_history: List[ValidationResult] = []
        self._issue_patterns: Dict[str, re.Pattern] = {}
        
        logger.info("Progress validator initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the validation system.
        """
        logger.info("Initializing progress validator")
        
        try:
            # Load configuration
            validation_config = await self.config_manager.get_setting('validation', {})
            self._validation_enabled = validation_config.get('enabled', True)
            self._auto_fix_enabled = validation_config.get('auto_fix_enabled', True)
            self._quality_threshold = validation_config.get('quality_threshold', 0.7)
            self._authenticity_threshold = validation_config.get('authenticity_threshold', 0.8)
            
            # Initialize validation components
            await self.placeholder_detector.initialize()
            await self.semantic_analyzer.initialize()
            await self.execution_tester.initialize()
            await self.auto_completion_engine.initialize()
            
            # Load issue patterns
            await self._load_issue_patterns()
            
            logger.info("Progress validator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize progress validator: {e}")
            raise
    
    async def validate_project(self, project: Project) -> ValidationResult:
        """
        Validate an entire project codebase.
        
        Args:
            project: Project to validate
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        logger.info(f"Validating project: {project.name}")
        
        start_time = datetime.now()
        
        try:
            if not self._validation_enabled:
                return ValidationResult(
                    is_valid=True,
                    overall_score=1.0,
                    authenticity_score=1.0,
                    completeness_score=1.0,
                    quality_score=1.0,
                    issues=[],
                    summary="Validation disabled",
                    execution_time=0.0,
                    validated_at=start_time
                )
            
            all_issues = []
            total_files = 0
            validated_files = 0
            
            # Validate all relevant files in project
            for file_path in self._get_project_files(project.path):
                total_files += 1
                
                try:
                    file_result = await self.validate_file_content(
                        file_path=file_path,
                        project=project
                    )\n                    \n                    all_issues.extend(file_result.issues)\n                    validated_files += 1\n                    \n                except Exception as e:\n                    logger.warning(f\"Failed to validate {file_path}: {e}\")\n                    all_issues.append(ValidationIssue(\n                        id=f\"validation_error_{file_path.name}\",\n                        description=f\"Validation failed: {e}\",\n                        severity=ValidationSeverity.MEDIUM,\n                        file_path=str(file_path),\n                        issue_type=\"validation_error\"\n                    ))\n            \n            # Calculate overall scores\n            scores = await self._calculate_project_scores(all_issues, project)\n            \n            # Determine if project is valid\n            is_valid = (\n                scores['overall_score'] >= self._quality_threshold and\n                scores['authenticity_score'] >= self._authenticity_threshold and\n                not any(issue.severity == ValidationSeverity.CRITICAL for issue in all_issues)\n            )\n            \n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            result = ValidationResult(\n                is_valid=is_valid,\n                overall_score=scores['overall_score'],\n                authenticity_score=scores['authenticity_score'],\n                completeness_score=scores['completeness_score'],\n                quality_score=scores['quality_score'],\n                issues=all_issues,\n                summary=self._generate_validation_summary(all_issues, scores),\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n            \n            # Store in history\n            self._validation_history.append(result)\n            \n            logger.info(\n                f\"Project validation completed: {validated_files}/{total_files} files, \"\n                f\"{len(all_issues)} issues, score: {scores['overall_score']:.2f}\"\n            )\n            \n            return result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"Project validation failed: {e}\")\n            \n            return ValidationResult(\n                is_valid=False,\n                overall_score=0.0,\n                authenticity_score=0.0,\n                completeness_score=0.0,\n                quality_score=0.0,\n                issues=[ValidationIssue(\n                    id=\"validation_failure\",\n                    description=f\"Validation system failure: {e}\",\n                    severity=ValidationSeverity.CRITICAL,\n                    issue_type=\"system_error\"\n                )],\n                summary=f\"Validation failed: {e}\",\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n    \n    async def validate_file_content(\n        self,\n        file_path: Path,\n        project: Optional[Project] = None,\n        content: Optional[str] = None\n    ) -> ValidationResult:\n        \"\"\"\n        Validate content of a single file.\n        \n        Args:\n            file_path: Path to file to validate\n            project: Associated project (optional)\n            content: File content (will read from file if None)\n            \n        Returns:\n            ValidationResult: File validation results\n        \"\"\"\n        start_time = datetime.now()\n        \n        try:\n            # Read content if not provided\n            if content is None:\n                if not file_path.exists():\n                    raise ValueError(f\"File does not exist: {file_path}\")\n                \n                with open(file_path, 'r', encoding='utf-8') as f:\n                    content = f.read()\n            \n            issues = []\n            \n            # Placeholder detection\n            placeholder_issues = await self.placeholder_detector.detect_placeholders(\n                content=content,\n                file_path=file_path\n            )\n            issues.extend(placeholder_issues)\n            \n            # Semantic analysis\n            semantic_issues = await self.semantic_analyzer.analyze_content(\n                content=content,\n                file_path=file_path,\n                project=project\n            )\n            issues.extend(semantic_issues)\n            \n            # Execution testing (for code files)\n            if self._is_executable_file(file_path):\n                execution_issues = await self.execution_tester.test_execution(\n                    content=content,\n                    file_path=file_path,\n                    project=project\n                )\n                issues.extend(execution_issues)\n            \n            # Pattern-based validation\n            pattern_issues = await self._pattern_based_validation(content, file_path)\n            issues.extend(pattern_issues)\n            \n            # Calculate scores\n            scores = await self._calculate_file_scores(content, issues, file_path)\n            \n            # Determine validity\n            is_valid = (\n                scores['overall_score'] >= self._quality_threshold and\n                not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)\n            )\n            \n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            result = ValidationResult(\n                is_valid=is_valid,\n                overall_score=scores['overall_score'],\n                authenticity_score=scores['authenticity_score'],\n                completeness_score=scores['completeness_score'],\n                quality_score=scores['quality_score'],\n                issues=issues,\n                summary=self._generate_file_summary(file_path, issues, scores),\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n            \n            return result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"File validation failed for {file_path}: {e}\")\n            \n            return ValidationResult(\n                is_valid=False,\n                overall_score=0.0,\n                authenticity_score=0.0,\n                completeness_score=0.0,\n                quality_score=0.0,\n                issues=[ValidationIssue(\n                    id=f\"file_validation_error_{file_path.name}\",\n                    description=f\"File validation failed: {e}\",\n                    severity=ValidationSeverity.HIGH,\n                    file_path=str(file_path),\n                    issue_type=\"validation_error\"\n                )],\n                summary=f\"Validation failed: {e}\",\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n    \n    async def validate_task_result(\n        self,\n        task: DevelopmentTask,\n        result: TaskResult,\n        project: Project\n    ) -> ValidationResult:\n        \"\"\"\n        Validate a task execution result.\n        \n        Args:\n            task: Original development task\n            result: Task execution result\n            project: Associated project\n            \n        Returns:\n            ValidationResult: Task result validation\n        \"\"\"\n        logger.debug(f\"Validating task result: {task.name}\")\n        \n        start_time = datetime.now()\n        \n        try:\n            # Validate generated content\n            content_result = await self.validate_generated_content(\n                content=result.generated_content or \"\",\n                task=task,\n                project=project\n            )\n            \n            # Add task-specific validation\n            task_issues = await self._validate_task_requirements(task, result)\n            content_result.issues.extend(task_issues)\n            \n            # Recalculate scores with task validation\n            scores = await self._calculate_task_scores(task, result, content_result.issues)\n            \n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            final_result = ValidationResult(\n                is_valid=content_result.is_valid and len(task_issues) == 0,\n                overall_score=scores['overall_score'],\n                authenticity_score=scores['authenticity_score'],\n                completeness_score=scores['completeness_score'],\n                quality_score=scores['quality_score'],\n                issues=content_result.issues,\n                summary=self._generate_task_summary(task, result, content_result.issues),\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n            \n            return final_result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"Task result validation failed: {e}\")\n            \n            return ValidationResult(\n                is_valid=False,\n                overall_score=0.0,\n                authenticity_score=0.0,\n                completeness_score=0.0,\n                quality_score=0.0,\n                issues=[],\n                summary=f\"Task validation failed: {e}\",\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n    \n    async def validate_generated_content(\n        self,\n        content: str,\n        task: Optional[DevelopmentTask] = None,\n        project: Optional[Project] = None\n    ) -> ValidationResult:\n        \"\"\"\n        Validate AI-generated content for quality and authenticity.\n        \n        Args:\n            content: Content to validate\n            task: Associated task (optional)\n            project: Associated project (optional)\n            \n        Returns:\n            ValidationResult: Content validation results\n        \"\"\"\n        logger.debug(\"Validating generated content\")\n        \n        start_time = datetime.now()\n        \n        try:\n            issues = []\n            \n            # Basic content validation\n            if not content.strip():\n                issues.append(ValidationIssue(\n                    id=\"empty_content\",\n                    description=\"Generated content is empty\",\n                    severity=ValidationSeverity.CRITICAL,\n                    issue_type=\"completeness\"\n                ))\n                \n                return ValidationResult(\n                    is_valid=False,\n                    overall_score=0.0,\n                    authenticity_score=0.0,\n                    completeness_score=0.0,\n                    quality_score=0.0,\n                    issues=issues,\n                    summary=\"Content is empty\",\n                    execution_time=0.0,\n                    validated_at=start_time\n                )\n            \n            # Placeholder detection\n            placeholder_issues = await self.placeholder_detector.detect_placeholders_in_content(\n                content=content\n            )\n            issues.extend(placeholder_issues)\n            \n            # Semantic validation\n            semantic_issues = await self.semantic_analyzer.analyze_generated_content(\n                content=content,\n                task=task,\n                project=project\n            )\n            issues.extend(semantic_issues)\n            \n            # Content quality analysis\n            quality_issues = await self._analyze_content_quality(content)\n            issues.extend(quality_issues)\n            \n            # Calculate scores\n            scores = await self._calculate_content_scores(content, issues)\n            \n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            result = ValidationResult(\n                is_valid=scores['overall_score'] >= self._quality_threshold,\n                overall_score=scores['overall_score'],\n                authenticity_score=scores['authenticity_score'],\n                completeness_score=scores['completeness_score'],\n                quality_score=scores['quality_score'],\n                issues=issues,\n                summary=self._generate_content_summary(content, issues),\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n            \n            return result\n            \n        except Exception as e:\n            execution_time = (datetime.now() - start_time).total_seconds()\n            logger.error(f\"Content validation failed: {e}\")\n            \n            return ValidationResult(\n                is_valid=False,\n                overall_score=0.0,\n                authenticity_score=0.0,\n                completeness_score=0.0,\n                quality_score=0.0,\n                issues=[],\n                summary=f\"Content validation failed: {e}\",\n                execution_time=execution_time,\n                validated_at=start_time\n            )\n    \n    async def auto_fix_issue(\n        self,\n        issue: ValidationIssue,\n        project: Project,\n        content: Optional[str] = None\n    ) -> Optional[str]:\n        \"\"\"\n        Attempt to automatically fix a validation issue.\n        \n        Args:\n            issue: Validation issue to fix\n            project: Associated project\n            content: Content to fix (optional)\n            \n        Returns:\n            str: Fixed content if successful, None otherwise\n        \"\"\"\n        if not self._auto_fix_enabled or not issue.auto_fixable:\n            return None\n        \n        logger.info(f\"Attempting to auto-fix issue: {issue.id}\")\n        \n        try:\n            # Use auto-completion engine to fix the issue\n            fixed_content = await self.auto_completion_engine.fix_issue(\n                issue=issue,\n                content=content,\n                project=project\n            )\n            \n            if fixed_content:\n                logger.info(f\"Successfully auto-fixed issue: {issue.id}\")\n                return fixed_content\n            else:\n                logger.warning(f\"Failed to auto-fix issue: {issue.id}\")\n                return None\n                \n        except Exception as e:\n            logger.error(f\"Auto-fix failed for issue {issue.id}: {e}\")\n            return None\n    \n    async def cleanup(self) -> None:\n        \"\"\"\n        Cleanup validation resources.\n        \"\"\"\n        logger.info(\"Cleaning up progress validator\")\n        \n        # Cleanup validation components\n        await self.placeholder_detector.cleanup()\n        await self.semantic_analyzer.cleanup()\n        await self.execution_tester.cleanup()\n        await self.auto_completion_engine.cleanup()\n        \n        # Clear history\n        self._validation_history.clear()\n        \n        logger.info(\"Progress validator cleanup completed\")\n    \n    # Private helper methods\n    \n    def _get_project_files(self, project_path: Path) -> List[Path]:\n        \"\"\"\n        Get all relevant files in project for validation.\n        \"\"\"\n        extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.yaml', '.yml'}\n        \n        files = []\n        for ext in extensions:\n            files.extend(project_path.rglob(f'*{ext}'))\n        \n        # Filter out node_modules, __pycache__, etc.\n        filtered_files = []\n        for file_path in files:\n            if not any(part.startswith('.') or part in ['node_modules', '__pycache__', 'venv'] \n                      for part in file_path.parts):\n                filtered_files.append(file_path)\n        \n        return filtered_files\n    \n    def _is_executable_file(self, file_path: Path) -> bool:\n        \"\"\"\n        Check if file is executable code.\n        \"\"\"\n        executable_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}\n        return file_path.suffix.lower() in executable_extensions\n    \n    async def _load_issue_patterns(self) -> None:\n        \"\"\"\n        Load regex patterns for issue detection.\n        \"\"\"\n        patterns = {\n            'todo_placeholder': re.compile(r'#\\s*TODO|//\\s*TODO|/\\*\\s*TODO', re.IGNORECASE),\n            'fixme_placeholder': re.compile(r'#\\s*FIXME|//\\s*FIXME|/\\*\\s*FIXME', re.IGNORECASE),\n            'placeholder_text': re.compile(r'PLACEHOLDER|\\.\\.\\.|pass\\s*#\\s*TODO', re.IGNORECASE),\n            'incomplete_function': re.compile(r'def\\s+\\w+\\([^)]*\\):\\s*$|function\\s+\\w+\\([^)]*\\)\\s*{\\s*}', re.MULTILINE),\n            'empty_catch': re.compile(r'catch\\s*\\([^)]*\\)\\s*{\\s*}|except[^:]*:\\s*pass', re.MULTILINE)\n        }\n        \n        self._issue_patterns.update(patterns)\n    \n    async def _pattern_based_validation(\n        self,\n        content: str,\n        file_path: Path\n    ) -> List[ValidationIssue]:\n        \"\"\"\n        Validate content using regex patterns.\n        \"\"\"\n        issues = []\n        \n        for pattern_name, pattern in self._issue_patterns.items():\n            matches = pattern.finditer(content)\n            \n            for match in matches:\n                line_number = content[:match.start()].count('\\n') + 1\n                \n                issues.append(ValidationIssue(\n                    id=f\"{pattern_name}_{line_number}\",\n                    description=f\"Pattern detected: {pattern_name}\",\n                    severity=ValidationSeverity.MEDIUM,\n                    file_path=str(file_path),\n                    line_number=line_number,\n                    issue_type=\"pattern_match\",\n                    auto_fixable=True\n                ))\n        \n        return issues\n    \n    async def _calculate_project_scores(\n        self,\n        issues: List[ValidationIssue],\n        project: Project\n    ) -> Dict[str, float]:\n        \"\"\"\n        Calculate project-level validation scores.\n        \"\"\"\n        if not issues:\n            return {\n                'overall_score': 1.0,\n                'authenticity_score': 1.0,\n                'completeness_score': 1.0,\n                'quality_score': 1.0\n            }\n        \n        # Weight issues by severity\n        severity_weights = {\n            ValidationSeverity.LOW: 0.1,\n            ValidationSeverity.MEDIUM: 0.3,\n            ValidationSeverity.HIGH: 0.7,\n            ValidationSeverity.CRITICAL: 1.0\n        }\n        \n        total_weight = sum(severity_weights[issue.severity] for issue in issues)\n        max_possible_weight = len(issues) * severity_weights[ValidationSeverity.CRITICAL]\n        \n        base_score = max(0.0, 1.0 - (total_weight / max(max_possible_weight, 1.0)))\n        \n        # Calculate specific scores\n        authenticity_issues = [i for i in issues if i.issue_type in ['placeholder', 'pattern_match']]\n        authenticity_score = max(0.0, 1.0 - len(authenticity_issues) * 0.1)\n        \n        completeness_issues = [i for i in issues if i.issue_type in ['completeness', 'empty_content']]\n        completeness_score = max(0.0, 1.0 - len(completeness_issues) * 0.2)\n        \n        quality_issues = [i for i in issues if i.issue_type in ['quality', 'semantic']]\n        quality_score = max(0.0, 1.0 - len(quality_issues) * 0.15)\n        \n        return {\n            'overall_score': base_score,\n            'authenticity_score': authenticity_score,\n            'completeness_score': completeness_score,\n            'quality_score': quality_score\n        }\n    \n    async def _calculate_file_scores(\n        self,\n        content: str,\n        issues: List[ValidationIssue],\n        file_path: Path\n    ) -> Dict[str, float]:\n        \"\"\"\n        Calculate file-level validation scores.\n        \"\"\"\n        # Simplified scoring - use same logic as project scores\n        return await self._calculate_project_scores(issues, None)\n    \n    async def _calculate_task_scores(\n        self,\n        task: DevelopmentTask,\n        result: TaskResult,\n        issues: List[ValidationIssue]\n    ) -> Dict[str, float]:\n        \"\"\"\n        Calculate task-specific validation scores.\n        \"\"\"\n        base_scores = await self._calculate_project_scores(issues, None)\n        \n        # Adjust based on task success\n        if not result.success:\n            base_scores['overall_score'] *= 0.5\n        \n        return base_scores\n    \n    async def _calculate_content_scores(\n        self,\n        content: str,\n        issues: List[ValidationIssue]\n    ) -> Dict[str, float]:\n        \"\"\"\n        Calculate content-specific validation scores.\n        \"\"\"\n        base_scores = await self._calculate_project_scores(issues, None)\n        \n        # Adjust based on content length and complexity\n        content_length_bonus = min(len(content.split()) / 100, 0.1)  # Max 10% bonus\n        base_scores['quality_score'] = min(1.0, base_scores['quality_score'] + content_length_bonus)\n        \n        return base_scores\n    \n    async def _validate_task_requirements(\n        self,\n        task: DevelopmentTask,\n        result: TaskResult\n    ) -> List[ValidationIssue]:\n        \"\"\"\n        Validate task-specific requirements.\n        \"\"\"\n        issues = []\n        \n        # Check if task has specific requirements\n        if hasattr(task, 'requirements') and task.requirements:\n            for requirement in task.requirements:\n                # Simple requirement checking\n                if isinstance(requirement, str) and requirement.lower() not in (result.generated_content or \"\").lower():\n                    issues.append(ValidationIssue(\n                        id=f\"requirement_missing_{hash(requirement)}\",\n                        description=f\"Task requirement not met: {requirement}\",\n                        severity=ValidationSeverity.HIGH,\n                        issue_type=\"requirement\",\n                        auto_fixable=False\n                    ))\n        \n        return issues\n    \n    async def _analyze_content_quality(self, content: str) -> List[ValidationIssue]:\n        \"\"\"\n        Analyze general content quality.\n        \"\"\"\n        issues = []\n        \n        # Check for very short content\n        if len(content.strip()) < 10:\n            issues.append(ValidationIssue(\n                id=\"content_too_short\",\n                description=\"Generated content is very short\",\n                severity=ValidationSeverity.MEDIUM,\n                issue_type=\"quality\"\n            ))\n        \n        # Check for repetitive content\n        words = content.split()\n        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words\n            issues.append(ValidationIssue(\n                id=\"repetitive_content\",\n                description=\"Content appears repetitive\",\n                severity=ValidationSeverity.LOW,\n                issue_type=\"quality\"\n            ))\n        \n        return issues\n    \n    def _generate_validation_summary(\n        self,\n        issues: List[ValidationIssue],\n        scores: Dict[str, float]\n    ) -> str:\n        \"\"\"\n        Generate human-readable validation summary.\n        \"\"\"\n        if not issues:\n            return \"Validation passed - no issues found\"\n        \n        critical = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])\n        high = len([i for i in issues if i.severity == ValidationSeverity.HIGH])\n        medium = len([i for i in issues if i.severity == ValidationSeverity.MEDIUM])\n        low = len([i for i in issues if i.severity == ValidationSeverity.LOW])\n        \n        summary = f\"Found {len(issues)} issues: \"\n        if critical:\n            summary += f\"{critical} critical, \"\n        if high:\n            summary += f\"{high} high, \"\n        if medium:\n            summary += f\"{medium} medium, \"\n        if low:\n            summary += f\"{low} low\"\n        \n        summary += f\". Overall score: {scores['overall_score']:.2f}\"\n        \n        return summary.rstrip(\", \")\n    \n    def _generate_file_summary(\n        self,\n        file_path: Path,\n        issues: List[ValidationIssue],\n        scores: Dict[str, float]\n    ) -> str:\n        \"\"\"\n        Generate file-specific validation summary.\n        \"\"\"\n        base_summary = self._generate_validation_summary(issues, scores)\n        return f\"File {file_path.name}: {base_summary}\"\n    \n    def _generate_task_summary(\n        self,\n        task: DevelopmentTask,\n        result: TaskResult,\n        issues: List[ValidationIssue]\n    ) -> str:\n        \"\"\"\n        Generate task-specific validation summary.\n        \"\"\"\n        success_status = \"succeeded\" if result.success else \"failed\"\n        issue_count = len(issues)\n        \n        return f\"Task '{task.name}' {success_status} with {issue_count} validation issues\"\n    \n    def _generate_content_summary(\n        self,\n        content: str,\n        issues: List[ValidationIssue]\n    ) -> str:\n        \"\"\"\n        Generate content validation summary.\n        \"\"\"\n        word_count = len(content.split())\n        issue_count = len(issues)\n        \n        return f\"Content validation: {word_count} words, {issue_count} issues found\"