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

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.models.task import DevelopmentTask, TaskResult
# Import shared types to prevent circular imports
from src.claude_tui.validation.types import ValidationIssue, ValidationSeverity, ValidationResult, IssueCategory
# Removed to fix circular import - will import in method when needed
from src.claude_tui.validation.semantic_analyzer import SemanticAnalyzer
from src.claude_tui.validation.execution_tester import ExecutionTester
from src.claude_tui.validation.auto_completion_engine import AutoCompletionEngine

logger = logging.getLogger(__name__)


# ValidationSeverity, ValidationIssue, and ValidationResult now imported from types.py


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
        self.placeholder_detector = None  # Will be initialized when needed to avoid circular import
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
            # Initialize placeholder detector when needed to avoid circular import
            if self.placeholder_detector is None:
                from src.claude_tui.validation.placeholder_detector import PlaceholderDetector
                self.placeholder_detector = PlaceholderDetector(self.config_manager)
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
                    )
                except Exception as e:
                    self.logger.error(f"Failed to validate file {file_path}: {e}")
                    file_result = None
                
                if file_result:
                    file_results.append(file_result)
                    if file_result.get('has_issues'):
                        issues_found += 1
            
            # Calculate authenticity score
            authenticity_score = ((total_files - issues_found) / total_files * 100) if total_files > 0 else 0
            
            return {
                'authenticity_score': authenticity_score,
                'total_files': total_files,
                'issues_found': issues_found,
                'file_results': file_results,
                'validation_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Project validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'authenticity_score': 0,
                'error': str(e)
            }
    
    def _get_project_files(self, project_path: Path) -> List[Path]:
        """Get all relevant files in project for validation."""
        files = []
        try:
            for pattern in ['**/*.py', '**/*.js', '**/*.ts', '**/*.jsx', '**/*.tsx']:
                files.extend(project_path.glob(pattern))
        except Exception as e:
            self.logger.error(f"Failed to get project files: {e}")
        return files[:100]  # Limit to first 100 files for performance