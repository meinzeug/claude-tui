"""
Validation system shared types and enums.
Extracted to prevent circular imports between validation modules.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from pathlib import Path


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class PlaceholderType(Enum):
    """Types of placeholder content that can be detected."""
    TODO_COMMENT = auto()
    FIXME_COMMENT = auto()
    NOT_IMPLEMENTED = auto()
    GENERIC_ERROR = auto()
    EMPTY_FUNCTION = auto()
    MOCK_DATA = auto()
    EXAMPLE_CODE = auto()
    TEMPLATE_CODE = auto()


class IssueCategory(Enum):
    """Categories for validation issues."""
    PLACEHOLDER = "placeholder"
    INCOMPLETE = "incomplete"
    SYNTAX = "syntax"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in code."""
    
    # Core identification
    id: str
    category: IssueCategory
    severity: ValidationSeverity
    message: str
    description: str
    
    # Location information
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    
    # Context
    code_snippet: Optional[str] = None
    context_lines: Optional[List[str]] = None
    
    # Suggestions
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    
    # Metadata
    rule_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ValidationResult:
    """Complete validation results for code analysis."""
    
    # Overall status
    is_valid: bool
    is_complete: bool
    is_authentic: bool
    
    # Issues found
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Statistics
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    # Quality metrics
    quality_score: float = 0.0
    completeness_score: float = 0.0
    authenticity_score: float = 0.0
    
    # Placeholder analysis
    placeholders_found: int = 0
    placeholder_density: float = 0.0
    todos_found: int = 0
    fixmes_found: int = 0
    
    # Suggestions
    completion_suggestions: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Context
    file_path: Optional[Path] = None
    analysis_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived statistics after initialization."""
        self.total_issues = len(self.issues)
        
        # Count issues by severity
        severity_counts = {
            ValidationSeverity.CRITICAL: 0,
            ValidationSeverity.HIGH: 0,
            ValidationSeverity.MEDIUM: 0,
            ValidationSeverity.LOW: 0
        }
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        self.critical_issues = severity_counts[ValidationSeverity.CRITICAL]
        self.high_issues = severity_counts[ValidationSeverity.HIGH]
        self.medium_issues = severity_counts[ValidationSeverity.MEDIUM]
        self.low_issues = severity_counts[ValidationSeverity.LOW]
        
        # Count placeholder-specific issues
        placeholder_issues = [i for i in self.issues if i.category == IssueCategory.PLACEHOLDER]
        self.placeholders_found = len(placeholder_issues)
        
        # Count specific placeholder types
        self.todos_found = len([i for i in placeholder_issues if "TODO" in i.message.upper()])
        self.fixmes_found = len([i for i in placeholder_issues if "FIXME" in i.message.upper()])
        
        # Calculate overall scores (can be overridden by specific analyzers)
        if self.quality_score == 0.0:
            self._calculate_quality_score()
    
    def _calculate_quality_score(self):
        """Calculate a basic quality score based on issues found."""
        if self.total_issues == 0:
            self.quality_score = 10.0
            return
        
        # Weight issues by severity
        penalty = (
            self.critical_issues * 4.0 +
            self.high_issues * 2.0 + 
            self.medium_issues * 1.0 +
            self.low_issues * 0.5
        )
        
        # Base score of 10, subtract penalties
        self.quality_score = max(0.0, 10.0 - penalty)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a new issue and recalculate statistics."""
        self.issues.append(issue)
        self.__post_init__()  # Recalculate stats
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return self.critical_issues > 0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: IssueCategory) -> List[ValidationIssue]:
        """Get all issues of a specific category."""
        return [issue for issue in self.issues if issue.category == category]


@dataclass
class PlaceholderPattern:
    """Pattern definition for detecting placeholder content."""
    
    name: str
    pattern: Any  # re.Pattern or string
    placeholder_type: PlaceholderType
    severity: ValidationSeverity
    language: Optional[str] = None
    description: Optional[str] = None
    auto_fixable: bool = False
    suggested_replacement: Optional[str] = None
    confidence: float = 1.0