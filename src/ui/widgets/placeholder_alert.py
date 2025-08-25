#!/usr/bin/env python3
"""
Placeholder Alert Widget - Anti-hallucination warning system
for detecting and alerting about placeholder code, TODOs, and incomplete implementations.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Label, Button, ListView, ListItem
from textual.screen import ModalScreen
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console


class PlaceholderSeverity(Enum):
    """Severity levels for placeholder issues"""
    LOW = "low"          # Minor TODOs, comments
    MEDIUM = "medium"    # Empty functions, basic stubs
    HIGH = "high"        # Core functionality missing
    CRITICAL = "critical" # System-breaking placeholders


class PlaceholderType(Enum):
    """Types of placeholder patterns"""
    TODO_COMMENT = "todo_comment"
    EMPTY_FUNCTION = "empty_function"
    STUB_IMPLEMENTATION = "stub_implementation"
    MOCK_DATA = "mock_data"
    CONSOLE_LOG = "console_log"
    NOT_IMPLEMENTED = "not_implemented"
    PLACEHOLDER_TEXT = "placeholder_text"
    DEMO_CODE = "demo_code"
    HARDCODED_VALUE = "hardcoded_value"
    INCOMPLETE_LOGIC = "incomplete_logic"


@dataclass
class PlaceholderIssue:
    """Detected placeholder issue"""
    file_path: str
    line_number: int
    column: int
    issue_type: PlaceholderType
    severity: PlaceholderSeverity
    description: str
    code_snippet: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    context_lines: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.context_lines is None:
            self.context_lines = []
    
    @property
    def location(self) -> str:
        """Get formatted location string"""
        return f"{self.file_path}:{self.line_number}:{self.column}"
    
    def get_severity_icon(self) -> str:
        """Get icon for severity level"""
        icons = {
            PlaceholderSeverity.LOW: "🟡",
            PlaceholderSeverity.MEDIUM: "🟠",
            PlaceholderSeverity.HIGH: "🟠",
            PlaceholderSeverity.CRITICAL: "🔴"
        }
        return icons.get(self.severity, "⚪")
    
    def get_type_icon(self) -> str:
        """Get icon for issue type"""
        icons = {
            PlaceholderType.TODO_COMMENT: "📋",
            PlaceholderType.EMPTY_FUNCTION: "🕳️",
            PlaceholderType.STUB_IMPLEMENTATION: "🔨",
            PlaceholderType.MOCK_DATA: "🎭",
            PlaceholderType.CONSOLE_LOG: "📝",
            PlaceholderType.NOT_IMPLEMENTED: "❌",
            PlaceholderType.PLACEHOLDER_TEXT: "📄",
            PlaceholderType.DEMO_CODE: "🎆",
            PlaceholderType.HARDCODED_VALUE: "🔒",
            PlaceholderType.INCOMPLETE_LOGIC: "🧩"
        }
        return icons.get(self.issue_type, "❓")


class PlaceholderIssueWidget(Container):
    """Widget displaying individual placeholder issue"""
    
    def __init__(self, issue: PlaceholderIssue) -> None:
        super().__init__()
        self.issue = issue
        
    def compose(self) -> ComposeResult:
        """Compose issue widget"""
        with Vertical():
            # Issue header
            header_text = (
                f"{self.issue.get_severity_icon()} {self.issue.get_type_icon()} "
                f"{self.issue.description}"
            )
            yield Label(header_text, classes="issue-header")
            
            # Location
            yield Label(f"📁 {self.issue.location}", classes="issue-location")
            
            # Code snippet with syntax highlighting
            if self.issue.code_snippet:
                # Detect language from file extension
                language = self._detect_language(self.issue.file_path)
                
                try:
                    syntax = Syntax(
                        self.issue.code_snippet,
                        language,
                        theme="monokai",
                        line_numbers=True,
                        start_line=max(1, self.issue.line_number - 2)
                    )
                    yield Static(syntax, classes="code-snippet")
                except:
                    # Fallback to plain text
                    yield Static(self.issue.code_snippet, classes="code-snippet-plain")
            
            # Suggestion if available
            if self.issue.suggestion:
                suggestion_text = Text()
                suggestion_text.append("💡 Suggestion: ", style="bold yellow")
                suggestion_text.append(self.issue.suggestion)
                yield Static(suggestion_text, classes="suggestion")
            
            # Action buttons
            with Horizontal(classes="issue-actions"):
                if self.issue.auto_fixable:
                    yield Button("🔧 Auto Fix", id=f"fix-{id(self.issue)}")
                
                yield Button("👁️ View File", id=f"view-{id(self.issue)}")
                yield Button("❌ Ignore", id=f"ignore-{id(self.issue)}")
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.sh': 'bash',
            '.sql': 'sql',
        }
        
        for ext, lang in extension_map.items():
            if file_path.endswith(ext):
                return lang
        
        return 'text'


class PlaceholderAlertModal(ModalScreen):
    """Modal screen for displaying placeholder alerts"""
    
    def __init__(self, issues: List[PlaceholderIssue]) -> None:
        super().__init__()
        self.issues = issues
        self.issue_widgets: List[PlaceholderIssueWidget] = []
        
    def compose(self) -> ComposeResult:
        """Compose alert modal"""
        with Container(id="alert-modal"):
            # Header
            header_text = f"🚨 PLACEHOLDER CODE DETECTED ({len(self.issues)} issues found)"
            yield Label(header_text, classes="alert-header")
            
            # Severity summary
            severity_summary = self._generate_severity_summary()
            yield Static(severity_summary, classes="severity-summary")
            
            # Issues list
            with ListView(id="issues-list"):
                for issue in self.issues:
                    issue_widget = PlaceholderIssueWidget(issue)
                    self.issue_widgets.append(issue_widget)
                    yield ListItem(issue_widget)
            
            # Action buttons
            with Horizontal(classes="modal-actions"):
                yield Button("🔧 Auto-Fix All", id="auto-fix-all", variant="primary")
                yield Button("🚀 Start Completion", id="start-completion", variant="success")
                yield Button("📄 Export Report", id="export-report")
                yield Button("❌ Close", id="close-modal", variant="error")
    
    def _generate_severity_summary(self) -> Table:
        """Generate severity summary table"""
        # Count issues by severity
        severity_counts = {
            PlaceholderSeverity.CRITICAL: 0,
            PlaceholderSeverity.HIGH: 0,
            PlaceholderSeverity.MEDIUM: 0,
            PlaceholderSeverity.LOW: 0
        }
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        # Create summary table
        table = Table("Severity", "Count", "Description", show_header=True)
        
        severity_info = {
            PlaceholderSeverity.CRITICAL: ("red", "System-breaking placeholders"),
            PlaceholderSeverity.HIGH: ("orange", "Core functionality missing"),
            PlaceholderSeverity.MEDIUM: ("yellow", "Empty functions, basic stubs"),
            PlaceholderSeverity.LOW: ("blue", "Minor TODOs, comments")
        }
        
        for severity, count in severity_counts.items():
            if count > 0:
                color, description = severity_info[severity]
                # Get icon based on severity level
                icon_map = {
                    PlaceholderSeverity.CRITICAL: "🔴",
                    PlaceholderSeverity.HIGH: "🟡", 
                    PlaceholderSeverity.MEDIUM: "🟠",
                    PlaceholderSeverity.LOW: "🔵"
                }
                icon = icon_map.get(severity, "❓")
                
                table.add_row(
                    f"{icon} {severity.value.title()}",
                    str(count),
                    description,
                    style=color
                )
        
        return table
    
    @on(Button.Pressed, "#auto-fix-all")
    def auto_fix_all_issues(self) -> None:
        """Auto-fix all fixable issues"""
        fixable_issues = [issue for issue in self.issues if issue.auto_fixable]
        if fixable_issues:
            self.post_message(AutoFixIssuesMessage(fixable_issues))
            self.dismiss()
        else:
            # Show no fixable issues message
            pass
    
    @on(Button.Pressed, "#start-completion")
    def start_completion_workflow(self) -> None:
        """Start AI completion workflow"""
        self.post_message(StartCompletionMessage(self.issues))
        self.dismiss()
    
    @on(Button.Pressed, "#export-report")
    def export_report(self) -> None:
        """Export placeholder report"""
        self.post_message(ExportPlaceholderReportMessage(self.issues))
    
    @on(Button.Pressed, "#close-modal")
    def close_modal(self) -> None:
        """Close the modal"""
        self.dismiss()


class PlaceholderAlert(Container):
    """Main placeholder alert system widget"""
    
    current_issues: reactive[List[PlaceholderIssue]] = reactive([])
    alert_threshold: reactive[int] = reactive(3)  # Show alert if >3 issues
    
    def __init__(self) -> None:
        super().__init__()
        self.alert_visible = False
        self.monitoring_active = False
        
    def compose(self) -> ComposeResult:
        """Compose placeholder alert widget (initially hidden)"""
        # This widget is primarily for managing alerts, not visible UI
        yield Static("", id="placeholder-alert-manager")
    
    def on_mount(self) -> None:
        """Start monitoring when mounted"""
        self.start_monitoring()
    
    def watch_current_issues(self, issues: List[PlaceholderIssue]) -> None:
        """React to changes in current issues"""
        if len(issues) >= self.alert_threshold:
            self.show_alert(issues)
        elif self.alert_visible and len(issues) == 0:
            self.hide_alert()
    
    def show_alert(self, issues: List[PlaceholderIssue]) -> None:
        """Show placeholder alert modal"""
        if not self.alert_visible:
            self.alert_visible = True
            
            # Create and push alert modal
            alert_modal = PlaceholderAlertModal(issues)
            self.app.push_screen(alert_modal)
            
            # Send notification
            self.post_message(PlaceholderAlertTriggeredMessage(issues))
    
    def hide_alert(self) -> None:
        """Hide placeholder alert"""
        self.alert_visible = False
    
    @work(exclusive=True)
    async def start_monitoring(self) -> None:
        """Start continuous placeholder monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # This would integrate with the validation engine
                # For now, simulate detection
                await self._check_for_placeholders()
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                # Log error and continue monitoring
                await asyncio.sleep(120)  # Wait longer on errors
    
    async def _check_for_placeholders(self) -> None:
        """Check for placeholder issues in the current project"""
        # This would be replaced with actual validation engine integration
        # For development, create sample issues when needed
        
        # Simulate finding issues occasionally for testing
        import random
        if random.random() < 0.1:  # 10% chance of finding issues
            sample_issues = self._create_sample_issues()
            self.current_issues = sample_issues
    
    def _create_sample_issues(self) -> List[PlaceholderIssue]:
        """Create sample issues for development/testing"""
        sample_issues = [
            PlaceholderIssue(
                file_path="src/auth/authentication.py",
                line_number=45,
                column=1,
                issue_type=PlaceholderType.EMPTY_FUNCTION,
                severity=PlaceholderSeverity.HIGH,
                description="Empty authenticate() function found",
                code_snippet="def authenticate(username, password):\n    # TODO: implement authentication logic\n    pass",
                suggestion="Implement proper authentication with password hashing and session management",
                auto_fixable=False
            ),
            PlaceholderIssue(
                file_path="src/api/endpoints.py",
                line_number=23,
                column=5,
                issue_type=PlaceholderType.MOCK_DATA,
                severity=PlaceholderSeverity.MEDIUM,
                description="Mock data used instead of real API call",
                code_snippet="def get_user_data():\n    return {'id': 1, 'name': 'Test User'}  # Mock data",
                suggestion="Replace mock data with actual database query or API call",
                auto_fixable=True
            ),
            PlaceholderIssue(
                file_path="src/utils/helpers.py",
                line_number=12,
                column=1,
                issue_type=PlaceholderType.TODO_COMMENT,
                severity=PlaceholderSeverity.LOW,
                description="TODO comment indicates incomplete implementation",
                code_snippet="# TODO: Add input validation",
                suggestion="Add comprehensive input validation for all user inputs",
                auto_fixable=False
            )
        ]
        
        return sample_issues
    
    def trigger_immediate_scan(self) -> None:
        """Trigger immediate placeholder scan"""
        asyncio.create_task(self._check_for_placeholders())
    
    def update_alert_threshold(self, threshold: int) -> None:
        """Update alert threshold"""
        self.alert_threshold = threshold
    
    def add_custom_issue(self, issue: PlaceholderIssue) -> None:
        """Add custom placeholder issue"""
        current = list(self.current_issues)
        current.append(issue)
        self.current_issues = current
    
    def resolve_issue(self, issue: PlaceholderIssue) -> None:
        """Mark issue as resolved"""
        current = list(self.current_issues)
        if issue in current:
            current.remove(issue)
            self.current_issues = current
    
    def stop_monitoring(self) -> None:
        """Stop placeholder monitoring"""
        self.monitoring_active = False


class PlaceholderAlertTriggeredMessage(Message):
    """Message sent when placeholder alert is triggered"""
    
    def __init__(self, issues: List[PlaceholderIssue]) -> None:
        super().__init__()
        self.issues = issues


class AutoFixIssuesMessage(Message):
    """Message to auto-fix placeholder issues"""
    
    def __init__(self, issues: List[PlaceholderIssue]) -> None:
        super().__init__()
        self.issues = issues


class StartCompletionMessage(Message):
    """Message to start AI completion workflow"""
    
    def __init__(self, issues: List[PlaceholderIssue]) -> None:
        super().__init__()
        self.issues = issues


class ExportPlaceholderReportMessage(Message):
    """Message to export placeholder report"""
    
    def __init__(self, issues: List[PlaceholderIssue]) -> None:
        super().__init__()
        self.issues = issues