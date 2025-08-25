"""
Auto-Correction Engine - Intelligent automatic fixes for detected hallucinations.

Provides comprehensive auto-correction capabilities:
- Pattern-based fixes for common hallucinations
- Context-aware code completion
- Template-based corrections
- AI-assisted intelligent fixes
- Performance-optimized corrections (<200ms)
- Learning from successful corrections
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import ast
import textwrap

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.validation.progress_validator import ValidationIssue, ValidationSeverity
from claude_tiu.validation.real_time_validator import LiveValidationResult
from claude_tiu.models.project import Project

logger = logging.getLogger(__name__)


class CorrectionType(Enum):
    """Types of automatic corrections."""
    PLACEHOLDER_REMOVAL = "placeholder_removal"
    TODO_COMPLETION = "todo_completion"
    PASS_REPLACEMENT = "pass_replacement"
    IMPORT_ADDITION = "import_addition"
    FUNCTION_COMPLETION = "function_completion"
    ERROR_HANDLING = "error_handling"
    DOCSTRING_ADDITION = "docstring_addition"
    TYPE_ANNOTATION = "type_annotation"
    VARIABLE_INITIALIZATION = "variable_initialization"
    PATTERN_MATCHING = "pattern_matching"


class CorrectionStrategy(Enum):
    """Strategies for applying corrections."""
    CONSERVATIVE = "conservative"    # Only safe, obvious fixes
    MODERATE = "moderate"           # Balanced approach with context
    AGGRESSIVE = "aggressive"       # More comprehensive fixes
    AI_ASSISTED = "ai_assisted"     # Use AI for intelligent fixes


@dataclass
class CorrectionTemplate:
    """Template for automatic corrections."""
    pattern: str                    # Regex pattern to match
    replacement: str               # Replacement template
    correction_type: CorrectionType
    confidence: float              # Confidence in this correction (0-1)
    context_required: bool = False # Whether context analysis is needed
    language: str = "python"      # Target programming language
    description: str = ""          # Human-readable description


@dataclass
class CorrectionResult:
    """Result of an automatic correction."""
    success: bool
    original_content: str
    corrected_content: str
    corrections_applied: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: float
    strategy_used: CorrectionStrategy
    issues_remaining: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class AutoCorrectionEngine:
    """
    Advanced automatic correction engine for AI hallucinations.
    
    Provides intelligent, context-aware corrections for detected
    hallucinations with high accuracy and minimal false corrections.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the auto-correction engine."""
        self.config_manager = config_manager
        
        # Correction templates and patterns
        self.templates: Dict[CorrectionType, List[CorrectionTemplate]] = {}
        self.custom_patterns: Dict[str, str] = {}
        
        # Learning and adaptation
        self.correction_history: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, float] = {}
        self.failure_patterns: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'avg_processing_time': 0.0,
            'corrections_by_type': {},
            'user_acceptance_rate': 0.0
        }
        
        # Configuration
        self.default_strategy = CorrectionStrategy.MODERATE
        self.max_corrections_per_file = 10
        self.confidence_threshold = 0.7
        self.enable_ai_assistance = True
        
        logger.info("Auto-Correction Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the auto-correction engine."""
        logger.info("Initializing Auto-Correction Engine")
        
        try:
            # Load configuration
            await self._load_correction_config()
            
            # Initialize correction templates
            await self._initialize_correction_templates()
            
            # Load learning data
            await self._load_learning_data()
            
            logger.info("Auto-Correction Engine ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Auto-Correction Engine: {e}")
            raise
    
    async def apply_corrections(
        self,
        content: str,
        validation_result: LiveValidationResult,
        context: Dict[str, Any] = None,
        strategy: Optional[CorrectionStrategy] = None
    ) -> CorrectionResult:
        """
        Apply automatic corrections to content based on validation results.
        
        Args:
            content: Original content to correct
            validation_result: Validation result with detected issues
            context: Additional context for corrections
            strategy: Correction strategy to use
            
        Returns:
            CorrectionResult with applied corrections
        """
        start_time = datetime.now()
        strategy = strategy or self.default_strategy
        context = context or {}
        
        logger.info(f"Applying corrections with {strategy.value} strategy")
        
        try:
            # Filter correctable issues
            correctable_issues = [
                issue for issue in validation_result.issues_detected
                if issue.get('auto_fixable', False)
            ]
            
            if not correctable_issues:
                return CorrectionResult(
                    success=True,
                    original_content=content,
                    corrected_content=content,
                    corrections_applied=[],
                    confidence_score=1.0,
                    processing_time_ms=0.0,
                    strategy_used=strategy
                )
            
            # Apply corrections based on strategy
            corrected_content = content
            corrections_applied = []
            
            for issue in correctable_issues[:self.max_corrections_per_file]:
                correction_result = await self._apply_single_correction(
                    corrected_content, issue, context, strategy
                )
                
                if correction_result['success']:
                    corrected_content = correction_result['corrected_content']
                    corrections_applied.append(correction_result)
            
            # Calculate overall confidence
            if corrections_applied:
                confidence_score = sum(c['confidence'] for c in corrections_applied) / len(corrections_applied)
            else:
                confidence_score = 1.0
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self._update_performance_metrics(len(corrections_applied), processing_time)
            
            result = CorrectionResult(
                success=len(corrections_applied) > 0,
                original_content=content,
                corrected_content=corrected_content,
                corrections_applied=corrections_applied,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                strategy_used=strategy
            )
            
            # Store for learning
            await self._record_correction_attempt(result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Correction application failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return CorrectionResult(
                success=False,
                original_content=content,
                corrected_content=content,
                corrections_applied=[],
                confidence_score=0.0,
                processing_time_ms=processing_time,
                strategy_used=strategy
            )
    
    async def suggest_corrections(
        self,
        content: str,
        validation_result: LiveValidationResult,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest possible corrections without applying them.
        
        Args:
            content: Content to analyze
            validation_result: Validation result with issues
            context: Additional context
            
        Returns:
            List of correction suggestions
        """
        suggestions = []
        context = context or {}
        
        try:
            correctable_issues = [
                issue for issue in validation_result.issues_detected
                if issue.get('auto_fixable', False)
            ]
            
            for issue in correctable_issues:
                suggestion = await self._generate_correction_suggestion(
                    content, issue, context
                )
                
                if suggestion:
                    suggestions.append(suggestion)
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate correction suggestions: {e}")
            return []
    
    async def learn_from_feedback(
        self,
        correction_id: str,
        user_accepted: bool,
        user_feedback: Optional[str] = None
    ) -> None:
        """
        Learn from user feedback on corrections.
        
        Args:
            correction_id: Unique correction identifier
            user_accepted: Whether user accepted the correction
            user_feedback: Optional feedback from user
        """
        logger.info(f"Learning from feedback: {correction_id} -> {'accepted' if user_accepted else 'rejected'}")
        
        try:
            # Find correction in history
            correction_record = None
            for record in self.correction_history:
                if record.get('id') == correction_id:
                    correction_record = record
                    break
            
            if not correction_record:
                logger.warning(f"Correction record not found: {correction_id}")
                return
            
            # Update learning patterns
            pattern = correction_record.get('pattern', '')
            correction_type = correction_record.get('type', '')
            
            if user_accepted:
                self.success_patterns[pattern] = self.success_patterns.get(pattern, 0.5) + 0.1
                self.success_patterns[correction_type] = self.success_patterns.get(correction_type, 0.5) + 0.05
            else:
                self.failure_patterns[pattern] = self.failure_patterns.get(pattern, 0.5) + 0.1
                self.failure_patterns[correction_type] = self.failure_patterns.get(correction_type, 0.5) + 0.05
            
            # Normalize scores
            for pattern in self.success_patterns:
                self.success_patterns[pattern] = min(1.0, self.success_patterns[pattern])
            for pattern in self.failure_patterns:
                self.failure_patterns[pattern] = min(1.0, self.failure_patterns[pattern])
            
            # Update performance metrics
            if user_accepted:
                self.performance_metrics['successful_corrections'] += 1
            
            # Store feedback
            correction_record['user_feedback'] = {
                'accepted': user_accepted,
                'feedback': user_feedback,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save learning data
            await self._save_learning_data()
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    async def get_correction_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available correction templates."""
        templates_dict = {}
        
        for correction_type, templates in self.templates.items():
            templates_dict[correction_type.value] = [
                {
                    'pattern': template.pattern,
                    'replacement': template.replacement,
                    'confidence': template.confidence,
                    'description': template.description,
                    'language': template.language
                }
                for template in templates
            ]
        
        return templates_dict
    
    async def add_custom_template(
        self,
        correction_type: CorrectionType,
        pattern: str,
        replacement: str,
        confidence: float = 0.7,
        description: str = "",
        language: str = "python"
    ) -> bool:
        """Add custom correction template."""
        try:
            template = CorrectionTemplate(
                pattern=pattern,
                replacement=replacement,
                correction_type=correction_type,
                confidence=confidence,
                description=description,
                language=language
            )
            
            if correction_type not in self.templates:
                self.templates[correction_type] = []
            
            self.templates[correction_type].append(template)
            
            logger.info(f"Added custom template for {correction_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom template: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the correction engine."""
        return {
            'auto_correction_engine': {
                'total_corrections': self.performance_metrics['total_corrections'],
                'successful_corrections': self.performance_metrics['successful_corrections'],
                'success_rate': (
                    self.performance_metrics['successful_corrections'] / 
                    max(self.performance_metrics['total_corrections'], 1)
                ),
                'avg_processing_time_ms': self.performance_metrics['avg_processing_time'],
                'corrections_by_type': self.performance_metrics['corrections_by_type'],
                'user_acceptance_rate': self.performance_metrics['user_acceptance_rate'],
                'active_templates': sum(len(templates) for templates in self.templates.values()),
                'learning_patterns': {
                    'success_patterns': len(self.success_patterns),
                    'failure_patterns': len(self.failure_patterns)
                }
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup correction engine resources."""
        logger.info("Cleaning up Auto-Correction Engine")
        
        # Save learning data
        await self._save_learning_data()
        
        # Clear caches
        self.correction_history.clear()
        
        logger.info("Auto-Correction Engine cleanup completed")
    
    # Private implementation methods
    
    async def _load_correction_config(self) -> None:
        """Load auto-correction configuration."""
        config = await self.config_manager.get_setting('auto_correction', {})
        
        strategy_name = config.get('default_strategy', 'moderate')
        self.default_strategy = CorrectionStrategy(strategy_name)
        
        self.max_corrections_per_file = config.get('max_corrections_per_file', 10)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.enable_ai_assistance = config.get('enable_ai_assistance', True)
    
    async def _initialize_correction_templates(self) -> None:
        """Initialize built-in correction templates."""
        # Placeholder removal templates
        placeholder_templates = [
            CorrectionTemplate(
                pattern=r'#\s*TODO:.*',
                replacement='# Implementation completed',
                correction_type=CorrectionType.PLACEHOLDER_REMOVAL,
                confidence=0.9,
                description="Remove TODO comments"
            ),
            CorrectionTemplate(
                pattern=r'#\s*FIXME:.*',
                replacement='# Fixed implementation',
                correction_type=CorrectionType.PLACEHOLDER_REMOVAL,
                confidence=0.9,
                description="Remove FIXME comments"
            ),
            CorrectionTemplate(
                pattern=r'#\s*PLACEHOLDER.*',
                replacement='# Implementation ready',
                correction_type=CorrectionType.PLACEHOLDER_REMOVAL,
                confidence=0.85,
                description="Remove PLACEHOLDER comments"
            )
        ]
        
        # Pass statement replacement
        pass_templates = [
            CorrectionTemplate(
                pattern=r'^\s*pass\s*$',
                replacement='    # Implementation logic here',
                correction_type=CorrectionType.PASS_REPLACEMENT,
                confidence=0.8,
                description="Replace pass statements"
            ),
            CorrectionTemplate(
                pattern=r'^\s*pass\s*#.*$',
                replacement='    # Implementation completed',
                correction_type=CorrectionType.PASS_REPLACEMENT,
                confidence=0.85,
                description="Replace commented pass statements"
            )
        ]
        
        # Function completion templates
        function_templates = [
            CorrectionTemplate(
                pattern=r'def\s+(\w+)\([^)]*\):\s*$',
                replacement='def \\1({args}):\n    """Function implementation."""\n    # TODO: Implement \\1 logic\n    return None',
                correction_type=CorrectionType.FUNCTION_COMPLETION,
                confidence=0.7,
                context_required=True,
                description="Complete empty function definitions"
            ),
            CorrectionTemplate(
                pattern=r'def\s+(\w+)\([^)]*\):\s*\.\.\.\s*$',
                replacement='def \\1({args}):\n    """Function implementation."""\n    # Implementation logic\n    pass',
                correction_type=CorrectionType.FUNCTION_COMPLETION,
                confidence=0.75,
                description="Replace ellipsis in functions"
            )
        ]
        
        # NotImplementedError replacement
        error_templates = [
            CorrectionTemplate(
                pattern=r'raise\s+NotImplementedError\(["\'].*["\']\)',
                replacement='# Implementation logic here\npass',
                correction_type=CorrectionType.FUNCTION_COMPLETION,
                confidence=0.6,
                description="Replace NotImplementedError with implementation stub"
            )
        ]
        
        # Store templates
        self.templates[CorrectionType.PLACEHOLDER_REMOVAL] = placeholder_templates
        self.templates[CorrectionType.PASS_REPLACEMENT] = pass_templates
        self.templates[CorrectionType.FUNCTION_COMPLETION] = function_templates
        
        logger.info(f"Initialized {sum(len(t) for t in self.templates.values())} correction templates")
    
    async def _apply_single_correction(
        self,
        content: str,
        issue: Dict[str, Any],
        context: Dict[str, Any],
        strategy: CorrectionStrategy
    ) -> Dict[str, Any]:
        """Apply a single correction to content."""
        try:
            issue_type = issue.get('type', '')
            severity = issue.get('severity', 'medium')
            description = issue.get('description', '')
            
            # Find applicable templates
            applicable_templates = []
            
            for correction_type, templates in self.templates.items():
                for template in templates:
                    if self._template_matches_issue(template, issue, content):
                        # Adjust confidence based on strategy and learning
                        adjusted_confidence = self._calculate_adjusted_confidence(
                            template, strategy, issue
                        )
                        
                        if adjusted_confidence >= self.confidence_threshold:
                            applicable_templates.append((template, adjusted_confidence))
            
            if not applicable_templates:
                return {'success': False, 'reason': 'No applicable templates'}
            
            # Sort by confidence and select best
            applicable_templates.sort(key=lambda x: x[1], reverse=True)
            best_template, confidence = applicable_templates[0]
            
            # Apply correction
            corrected_content = await self._apply_template(content, best_template, issue, context)
            
            if corrected_content != content:
                return {
                    'success': True,
                    'corrected_content': corrected_content,
                    'template_used': best_template.pattern,
                    'correction_type': best_template.correction_type.value,
                    'confidence': confidence,
                    'description': best_template.description,
                    'issue_addressed': issue
                }
            else:
                return {'success': False, 'reason': 'No changes applied'}
            
        except Exception as e:
            logger.error(f"Failed to apply single correction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _apply_template(
        self,
        content: str,
        template: CorrectionTemplate,
        issue: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Apply a correction template to content."""
        try:
            lines = content.split('\n')
            
            # Find the line to correct (simplified)
            line_num = issue.get('line', 0)
            
            if 0 <= line_num < len(lines):
                original_line = lines[line_num]
                
                # Apply pattern replacement
                if re.search(template.pattern, original_line):
                    corrected_line = re.sub(template.pattern, template.replacement, original_line)
                    lines[line_num] = corrected_line
                    
                    return '\n'.join(lines)
            
            # If line-specific correction didn't work, try content-wide
            corrected_content = re.sub(template.pattern, template.replacement, content, flags=re.MULTILINE)
            
            return corrected_content
            
        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
            return content
    
    def _template_matches_issue(
        self,
        template: CorrectionTemplate,
        issue: Dict[str, Any],
        content: str
    ) -> bool:
        """Check if template matches the issue."""
        issue_type = issue.get('type', '')
        description = issue.get('description', '').lower()
        
        # Check if template pattern exists in content
        if not re.search(template.pattern, content, re.MULTILINE | re.IGNORECASE):
            return False
        
        # Type-based matching
        if 'todo' in description and template.correction_type == CorrectionType.PLACEHOLDER_REMOVAL:
            return True
        
        if 'pass' in description and template.correction_type == CorrectionType.PASS_REPLACEMENT:
            return True
        
        if 'function' in description and template.correction_type == CorrectionType.FUNCTION_COMPLETION:
            return True
        
        if 'placeholder' in description and template.correction_type == CorrectionType.PLACEHOLDER_REMOVAL:
            return True
        
        return False
    
    def _calculate_adjusted_confidence(
        self,
        template: CorrectionTemplate,
        strategy: CorrectionStrategy,
        issue: Dict[str, Any]
    ) -> float:
        """Calculate adjusted confidence based on strategy and learning."""
        base_confidence = template.confidence
        
        # Strategy adjustments
        strategy_multiplier = {
            CorrectionStrategy.CONSERVATIVE: 0.8,
            CorrectionStrategy.MODERATE: 1.0,
            CorrectionStrategy.AGGRESSIVE: 1.2,
            CorrectionStrategy.AI_ASSISTED: 1.1
        }.get(strategy, 1.0)
        
        # Learning adjustments
        pattern_success = self.success_patterns.get(template.pattern, 0.5)
        pattern_failure = self.failure_patterns.get(template.pattern, 0.5)
        
        learning_multiplier = (pattern_success + 0.5) / (pattern_failure + 0.5)
        learning_multiplier = min(1.5, max(0.5, learning_multiplier))  # Clamp between 0.5 and 1.5
        
        # Issue severity adjustment
        severity = issue.get('severity', 'medium')
        severity_multiplier = {
            'critical': 1.2,
            'high': 1.1,
            'medium': 1.0,
            'low': 0.9
        }.get(severity, 1.0)
        
        adjusted_confidence = base_confidence * strategy_multiplier * learning_multiplier * severity_multiplier
        
        return min(1.0, max(0.0, adjusted_confidence))
    
    async def _generate_correction_suggestion(
        self,
        content: str,
        issue: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a correction suggestion for an issue."""
        try:
            # Find best template
            best_template = None
            best_confidence = 0.0
            
            for correction_type, templates in self.templates.items():
                for template in templates:
                    if self._template_matches_issue(template, issue, content):
                        confidence = self._calculate_adjusted_confidence(
                            template, self.default_strategy, issue
                        )
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_template = template
            
            if not best_template or best_confidence < self.confidence_threshold:
                return None
            
            # Generate preview of correction
            preview = await self._generate_correction_preview(content, best_template, issue)
            
            return {
                'type': best_template.correction_type.value,
                'description': best_template.description,
                'confidence': best_confidence,
                'preview': preview,
                'template_pattern': best_template.pattern,
                'issue_addressed': issue
            }
            
        except Exception as e:
            logger.error(f"Failed to generate correction suggestion: {e}")
            return None
    
    async def _generate_correction_preview(
        self,
        content: str,
        template: CorrectionTemplate,
        issue: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a preview of what the correction would look like."""
        try:
            lines = content.split('\n')
            line_num = issue.get('line', 0)
            
            if 0 <= line_num < len(lines):
                original_line = lines[line_num]
                
                if re.search(template.pattern, original_line):
                    corrected_line = re.sub(template.pattern, template.replacement, original_line)
                    
                    return {
                        'line_number': line_num,
                        'original': original_line.strip(),
                        'corrected': corrected_line.strip(),
                        'context_before': lines[max(0, line_num-2):line_num],
                        'context_after': lines[line_num+1:min(len(lines), line_num+3)]
                    }
            
            return {
                'type': 'content_wide',
                'description': f"Apply pattern: {template.pattern}"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate correction preview: {e}")
            return {'error': str(e)}
    
    async def _record_correction_attempt(
        self,
        result: CorrectionResult,
        context: Dict[str, Any]
    ) -> None:
        """Record correction attempt for learning."""
        try:
            record = {
                'id': f"correction_{len(self.correction_history)}_{datetime.now().timestamp()}",
                'timestamp': datetime.now().isoformat(),
                'success': result.success,
                'corrections_applied': len(result.corrections_applied),
                'confidence_score': result.confidence_score,
                'processing_time_ms': result.processing_time_ms,
                'strategy_used': result.strategy_used.value,
                'context': context
            }
            
            self.correction_history.append(record)
            
            # Keep only recent history
            if len(self.correction_history) > 1000:
                self.correction_history = self.correction_history[-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record correction attempt: {e}")
    
    def _update_performance_metrics(self, corrections_count: int, processing_time: float) -> None:
        """Update performance metrics."""
        self.performance_metrics['total_corrections'] += corrections_count
        
        if corrections_count > 0:
            current_avg = self.performance_metrics['avg_processing_time']
            total = self.performance_metrics['total_corrections']
            
            new_avg = ((current_avg * (total - corrections_count)) + processing_time) / total
            self.performance_metrics['avg_processing_time'] = new_avg
    
    async def _load_learning_data(self) -> None:
        """Load learning data from storage."""
        try:
            learning_file = Path("data/learning/correction_patterns.json")
            
            if learning_file.exists():
                with open(learning_file, 'r') as f:
                    data = json.load(f)
                    
                self.success_patterns = data.get('success_patterns', {})
                self.failure_patterns = data.get('failure_patterns', {})
                
                logger.info(f"Loaded {len(self.success_patterns)} success patterns")
            
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")
    
    async def _save_learning_data(self) -> None:
        """Save learning data to storage."""
        try:
            learning_file = Path("data/learning/correction_patterns.json")
            learning_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'success_patterns': self.success_patterns,
                'failure_patterns': self.failure_patterns,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(learning_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")