"""
Git Workflow Visualization Widget

This module provides comprehensive UI components for visualizing and managing
Git workflows within Claude-TIU, including:
- Real-time Git repository status
- Branch visualization and management
- Pull request dashboard
- Code review interface
- Conflict resolution assistant
- CI/CD pipeline status
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ListItem, ListView,
    ProgressBar, Static, Switch, Tab, TabbedContent, TabPane, Tree, Log
)
from textual.reactive import reactive, var
from textual.message import Message
from textual.binding import Binding

from ...integrations.git_advanced import (
    EnhancedGitManager, PullRequest, CodeReview, BranchRule,
    PRStatus, ReviewStatus, CIStatus
)
from ...core.types import ValidationResult

logger = logging.getLogger(__name__)


class GitStatusWidget(Static):
    """Widget displaying current Git repository status."""
    
    def __init__(self, git_manager: EnhancedGitManager, **kwargs):
        """Initialize Git status widget."""
        super().__init__(**kwargs)
        self.git_manager = git_manager
        self.refresh_interval = 10  # seconds
        self._update_task: Optional[asyncio.Task] = None
    
    def compose(self) -> ComposeResult:
        """Compose the Git status display."""
        with Container(classes="git-status-container"):
            yield Label("Git Repository Status", classes="section-header")
            
            with Grid(classes="git-status-grid"):
                with Vertical(classes="status-info"):
                    yield Label("Branch:", classes="info-label")
                    yield Label("", id="current-branch", classes="info-value")
                    
                    yield Label("Status:", classes="info-label")
                    yield Label("", id="repo-status", classes="info-value")
                    
                    yield Label("Changes:", classes="info-label")
                    yield Label("", id="changes-summary", classes="info-value")
                
                with Vertical(classes="file-status"):
                    yield Label("Modified Files", classes="subsection-header")
                    yield ListView(id="modified-files", classes="file-list")
                    
                    yield Label("Staged Files", classes="subsection-header")
                    yield ListView(id="staged-files", classes="file-list")
                    
                    yield Label("Untracked Files", classes="subsection-header")
                    yield ListView(id="untracked-files", classes="file-list")
    
    def on_mount(self) -> None:
        """Start automatic status updates when mounted."""
        self._update_task = asyncio.create_task(self._update_status_loop())
    
    def on_unmount(self) -> None:
        """Stop updates when unmounted."""
        if self._update_task:
            self._update_task.cancel()
    
    async def _update_status_loop(self) -> None:
        """Continuously update Git status."""
        while True:
            try:
                await self.update_status()
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Git status update failed: {e}")
                await asyncio.sleep(self.refresh_interval * 2)  # Back off on error
    
    async def update_status(self) -> None:
        """Update Git repository status display."""
        try:
            status = await self.git_manager.get_repository_status()
            
            # Update branch info
            branch_label = self.query_one("#current-branch", Label)
            branch_label.update(status.get('current_branch', 'Unknown'))
            
            # Update repository status
            repo_status_label = self.query_one("#repo-status", Label)
            if status.get('is_dirty', False):
                repo_status_label.update("Dirty (uncommitted changes)")
                repo_status_label.add_class("status-dirty")
            else:
                repo_status_label.update("Clean")
                repo_status_label.remove_class("status-dirty")
            
            # Update changes summary
            files = status.get('files', {})
            changes_summary = self.query_one("#changes-summary", Label)
            summary_text = f"M: {len(files.get('modified', []))}, S: {len(files.get('staged', []))}, U: {len(files.get('untracked', []))}"
            changes_summary.update(summary_text)
            
            # Update file lists
            await self._update_file_list("modified-files", files.get('modified', []))
            await self._update_file_list("staged-files", files.get('staged', []))
            await self._update_file_list("untracked-files", files.get('untracked', []))
            
        except Exception as e:
            logger.error(f"Failed to update Git status: {e}")
            # Show error state
            branch_label = self.query_one("#current-branch", Label)
            branch_label.update("Error")
    
    async def _update_file_list(self, list_id: str, files: List[str]) -> None:
        """Update a file list widget."""
        try:
            file_list = self.query_one(f"#{list_id}", ListView)
            file_list.clear()
            
            for file_path in files:
                # Truncate long paths for display
                display_path = file_path
                if len(display_path) > 40:
                    display_path = "..." + display_path[-37:]
                
                file_list.append(ListItem(Label(display_path)))
                
        except Exception as e:
            logger.warning(f"Failed to update file list {list_id}: {e}")


class BranchTreeWidget(Static):
    """Widget for visualizing and managing Git branches."""
    
    def __init__(self, git_manager: EnhancedGitManager, **kwargs):
        """Initialize branch tree widget."""
        super().__init__(**kwargs)
        self.git_manager = git_manager
    
    def compose(self) -> ComposeResult:
        """Compose the branch tree interface."""
        with Container(classes="branch-tree-container"):
            yield Label("Branch Management", classes="section-header")
            
            with Horizontal(classes="branch-controls"):
                yield Input(placeholder="New branch name", id="new-branch-input")
                yield Button("Create Branch", id="create-branch-btn", variant="primary")
                yield Button("Refresh", id="refresh-branches-btn")
            
            with Container(classes="branch-tree-content"):
                yield Tree("Branches", id="branch-tree")
                
                with Vertical(classes="branch-details"):
                    yield Label("Branch Details", classes="subsection-header")
                    yield Static("", id="branch-info", classes="branch-info-panel")
    
    def on_mount(self) -> None:
        """Initialize branch tree when mounted."""
        asyncio.create_task(self.refresh_branches())
    
    @on(Button.Pressed, "#create-branch-btn")
    async def create_new_branch(self) -> None:
        """Create a new branch."""
        branch_input = self.query_one("#new-branch-input", Input)
        branch_name = branch_input.value.strip()
        
        if not branch_name:
            self.notify("Please enter a branch name", severity="warning")
            return
        
        try:
            result = await self.git_manager.create_branch(
                branch_name=branch_name,
                checkout=True
            )
            
            if result.is_success:
                self.notify(f"Branch '{branch_name}' created successfully", severity="information")
                branch_input.value = ""
                await self.refresh_branches()
            else:
                self.notify(f"Failed to create branch: {result.message}", severity="error")
                
        except Exception as e:
            logger.error(f"Branch creation failed: {e}")
            self.notify(f"Branch creation failed: {e}", severity="error")
    
    @on(Button.Pressed, "#refresh-branches-btn")
    async def refresh_branches(self) -> None:
        """Refresh the branch tree display."""
        try:
            status = await self.git_manager.get_repository_status()
            branches = status.get('branches', [])
            
            tree = self.query_one("#branch-tree", Tree)
            tree.clear()
            
            # Organize branches by type
            local_branches = [b for b in branches if not b['is_remote']]
            remote_branches = [b for b in branches if b['is_remote']]
            
            # Add local branches
            if local_branches:
                local_node = tree.root.add("Local Branches")
                for branch in local_branches:
                    branch_name = branch['name']
                    if branch['is_current']:
                        branch_name += " (current)"
                    
                    branch_node = local_node.add(branch_name)
                    branch_node.data = branch
            
            # Add remote branches
            if remote_branches:
                remote_node = tree.root.add("Remote Branches")
                for branch in remote_branches:
                    branch_node = remote_node.add(branch['name'])
                    branch_node.data = branch
            
            tree.root.expand()
            
        except Exception as e:
            logger.error(f"Failed to refresh branches: {e}")
            self.notify("Failed to refresh branches", severity="error")
    
    @on(Tree.NodeSelected)
    async def show_branch_details(self, event: Tree.NodeSelected) -> None:
        """Show details for selected branch."""
        if hasattr(event.node, 'data') and event.node.data:
            branch_data = event.node.data
            
            details_panel = self.query_one("#branch-info", Static)
            
            details_html = f"""
            <div class="branch-details">
                <h3>{branch_data['name']}</h3>
                <p><strong>Type:</strong> {'Remote' if branch_data['is_remote'] else 'Local'}</p>
                <p><strong>Current:</strong> {'Yes' if branch_data.get('is_current') else 'No'}</p>
                <p><strong>Last Commit:</strong> {branch_data.get('last_commit_hash', 'N/A')}</p>
                <p><strong>Last Message:</strong> {branch_data.get('last_commit_message', 'N/A')[:50]}</p>
            </div>
            """
            
            details_panel.update(details_html)


class PullRequestDashboard(Static):
    """Dashboard for managing pull requests."""
    
    def __init__(self, git_manager: EnhancedGitManager, **kwargs):
        """Initialize PR dashboard."""
        super().__init__(**kwargs)
        self.git_manager = git_manager
        self.pull_requests: Dict[int, PullRequest] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the PR dashboard interface."""
        with Container(classes="pr-dashboard-container"):
            yield Label("Pull Request Dashboard", classes="section-header")
            
            with Horizontal(classes="pr-controls"):
                yield Button("Create PR", id="create-pr-btn", variant="primary")
                yield Button("Refresh", id="refresh-prs-btn")
                yield Switch(value=True, id="auto-review-switch")
                yield Label("Auto Review")
            
            with Container(classes="pr-content"):
                yield DataTable(id="pr-table", classes="pr-table")
                
                with Vertical(classes="pr-details"):
                    yield Label("PR Details", classes="subsection-header")
                    yield Static("", id="pr-details-panel", classes="pr-details-content")
    
    def on_mount(self) -> None:
        """Initialize PR dashboard when mounted."""
        self._setup_pr_table()
        asyncio.create_task(self.refresh_pull_requests())
    
    def _setup_pr_table(self) -> None:
        """Setup the PR data table."""
        table = self.query_one("#pr-table", DataTable)
        table.add_columns("PR #", "Title", "Status", "Branch", "Checks", "Reviews")
        table.cursor_type = "row"
    
    @on(Button.Pressed, "#create-pr-btn")
    async def create_pull_request(self) -> None:
        """Create a new pull request."""
        try:
            # Get current branch
            status = await self.git_manager.get_repository_status()
            current_branch = status.get('current_branch')
            
            if not current_branch or current_branch in ['main', 'master', 'develop']:
                self.notify("Please switch to a feature branch to create a PR", severity="warning")
                return
            
            # Create PR with AI-generated content
            pr = await self.git_manager.create_pull_request(
                head_branch=current_branch,
                base_branch="main",
                draft=False
            )
            
            self.pull_requests[pr.number] = pr
            self.notify(f"Pull request #{pr.number} created successfully", severity="information")
            await self.refresh_pull_requests()
            
        except Exception as e:
            logger.error(f"PR creation failed: {e}")
            self.notify(f"Failed to create PR: {e}", severity="error")
    
    @on(Button.Pressed, "#refresh-prs-btn")
    async def refresh_pull_requests(self) -> None:
        """Refresh the pull requests table."""
        try:
            table = self.query_one("#pr-table", DataTable)
            table.clear()
            
            # Add PRs to table
            for pr_number, pr in self.pull_requests.items():
                # Determine status styling
                status_text = pr.status.value.title()
                if pr.status == PRStatus.MERGED:
                    status_text = f"[green]{status_text}[/green]"
                elif pr.status == PRStatus.CLOSED:
                    status_text = f"[red]{status_text}[/red]"
                
                # Checks status
                checks_status = "✅" if pr.checks_passed else "❌" if pr.ci_status == CIStatus.FAILURE else "⏳"
                
                # Reviews status
                reviews_count = len(pr.reviews)
                reviews_status = f"{reviews_count} reviews"
                
                table.add_row(
                    str(pr.number),
                    pr.title[:30] + ("..." if len(pr.title) > 30 else ""),
                    status_text,
                    pr.head_branch,
                    checks_status,
                    reviews_status,
                    key=str(pr.number)
                )
            
        except Exception as e:
            logger.error(f"Failed to refresh PRs: {e}")
            self.notify("Failed to refresh pull requests", severity="error")
    
    @on(DataTable.RowSelected)
    async def show_pr_details(self, event: DataTable.RowSelected) -> None:
        """Show details for selected PR."""
        if event.row_key:
            pr_number = int(event.row_key.value)
            if pr_number in self.pull_requests:
                pr = self.pull_requests[pr_number]
                await self._display_pr_details(pr)
    
    async def _display_pr_details(self, pr: PullRequest) -> None:
        """Display detailed PR information."""
        details_panel = self.query_one("#pr-details-panel", Static)
        
        # Format PR details
        details_html = f"""
        <div class="pr-details">
            <h3>PR #{pr.number}: {pr.title}</h3>
            
            <div class="pr-meta">
                <p><strong>Status:</strong> {pr.status.value.title()}</p>
                <p><strong>Branch:</strong> {pr.head_branch} → {pr.base_branch}</p>
                <p><strong>Created:</strong> {pr.created_at.strftime('%Y-%m-%d %H:%M') if pr.created_at else 'Unknown'}</p>
                <p><strong>Mergeable:</strong> {'Yes' if pr.mergeable else 'No'}</p>
            </div>
            
            <div class="pr-description">
                <h4>Description</h4>
                <p>{pr.body[:200] + ('...' if len(pr.body) > 200 else '')}</p>
            </div>
            
            <div class="pr-checks">
                <h4>CI/CD Status</h4>
                <p>Status: {pr.ci_status.value.title()}</p>
                <p>Checks Passed: {'Yes' if pr.checks_passed else 'No'}</p>
            </div>
            
            <div class="pr-conflicts">
                <h4>Conflicts</h4>
                {"<p>No conflicts</p>" if not pr.conflicts else f"<p>{len(pr.conflicts)} conflicts detected</p>"}
            </div>
        </div>
        """
        
        details_panel.update(details_html)


class CodeReviewWidget(Static):
    """Widget for displaying and managing code reviews."""
    
    def __init__(self, git_manager: EnhancedGitManager, **kwargs):
        """Initialize code review widget."""
        super().__init__(**kwargs)
        self.git_manager = git_manager
        self.current_review: Optional[CodeReview] = None
    
    def compose(self) -> ComposeResult:
        """Compose the code review interface."""
        with Container(classes="code-review-container"):
            yield Label("Code Review", classes="section-header")
            
            with Horizontal(classes="review-controls"):
                yield Input(placeholder="PR number", id="pr-number-input", classes="pr-input")
                yield Button("Start Review", id="start-review-btn", variant="primary")
                yield Button("Quick Review", id="quick-review-btn")
                yield Button("Comprehensive Review", id="comprehensive-review-btn")
            
            with Container(classes="review-content"):
                with TabbedContent(id="review-tabs"):
                    with TabPane("Overview", id="review-overview"):
                        yield Static("", id="review-summary", classes="review-summary")
                    
                    with TabPane("Issues", id="review-issues"):
                        yield ListView(id="issues-list", classes="issues-list")
                    
                    with TabPane("Comments", id="review-comments"):
                        yield ListView(id="comments-list", classes="comments-list")
                    
                    with TabPane("Suggestions", id="review-suggestions"):
                        yield ListView(id="suggestions-list", classes="suggestions-list")
    
    @on(Button.Pressed, "#start-review-btn")
    async def start_review(self) -> None:
        """Start a standard code review."""
        await self._perform_review("standard")
    
    @on(Button.Pressed, "#quick-review-btn")
    async def quick_review(self) -> None:
        """Perform a quick code review."""
        await self._perform_review("quick")
    
    @on(Button.Pressed, "#comprehensive-review-btn")
    async def comprehensive_review(self) -> None:
        """Perform a comprehensive code review."""
        await self._perform_review("comprehensive")
    
    async def _perform_review(self, review_type: str) -> None:
        """Perform code review of specified type."""
        pr_input = self.query_one("#pr-number-input", Input)
        pr_number_str = pr_input.value.strip()
        
        if not pr_number_str:
            self.notify("Please enter a PR number", severity="warning")
            return
        
        try:
            pr_number = int(pr_number_str)
            
            self.notify(f"Starting {review_type} review for PR #{pr_number}...", severity="information")
            
            # Perform the review
            review = await self.git_manager.review_pull_request(
                pr_number=pr_number,
                review_type=review_type,
                auto_approve_safe=(review_type == "quick")
            )
            
            self.current_review = review
            await self._display_review_results(review)
            
            self.notify(f"Review completed: {review.status.value}", severity="information")
            
        except ValueError:
            self.notify("Please enter a valid PR number", severity="error")
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            self.notify(f"Review failed: {e}", severity="error")
    
    async def _display_review_results(self, review: CodeReview) -> None:
        """Display code review results."""
        # Update overview
        summary_panel = self.query_one("#review-summary", Static)
        summary_html = f"""
        <div class="review-summary">
            <h3>Review Summary</h3>
            <p><strong>PR:</strong> #{review.pull_request_id}</p>
            <p><strong>Reviewer:</strong> {review.reviewer}</p>
            <p><strong>Status:</strong> {review.status.value.replace('_', ' ').title()}</p>
            <p><strong>Quality Score:</strong> {review.quality_score:.1f}/100</p>
            <p><strong>Security Issues:</strong> {len(review.security_issues)}</p>
            <p><strong>Total Comments:</strong> {len(review.comments)}</p>
            <p><strong>AI Generated:</strong> {'Yes' if review.ai_generated else 'No'}</p>
        </div>
        """
        summary_panel.update(summary_html)
        
        # Update issues list
        issues_list = self.query_one("#issues-list", ListView)
        issues_list.clear()
        
        for issue in review.security_issues:
            severity_color = {
                "critical": "red",
                "high": "yellow", 
                "medium": "blue",
                "low": "gray"
            }.get(issue.severity.value, "white")
            
            issue_text = f"[{severity_color}]{issue.severity.value.upper()}[/{severity_color}]: {issue.description}"
            issues_list.append(ListItem(Label(issue_text)))
        
        # Update comments list
        comments_list = self.query_one("#comments-list", ListView)
        comments_list.clear()
        
        for comment in review.comments:
            comment_text = f"{comment.get('path', 'General')}: {comment.get('body', '')[:100]}..."
            comments_list.append(ListItem(Label(comment_text)))
        
        # Update suggestions list
        suggestions_list = self.query_one("#suggestions-list", ListView)
        suggestions_list.clear()
        
        for suggestion in review.suggestions:
            suggestions_list.append(ListItem(Label(suggestion)))


class ConflictResolutionWidget(Static):
    """Widget for resolving merge conflicts with AI assistance."""
    
    def __init__(self, git_manager: EnhancedGitManager, **kwargs):
        """Initialize conflict resolution widget."""
        super().__init__(**kwargs)
        self.git_manager = git_manager
        self.current_conflicts: List[str] = []
    
    def compose(self) -> ComposeResult:
        """Compose the conflict resolution interface."""
        with Container(classes="conflict-resolution-container"):
            yield Label("Conflict Resolution Assistant", classes="section-header")
            
            with Horizontal(classes="conflict-controls"):
                yield Button("Scan Conflicts", id="scan-conflicts-btn", variant="primary")
                yield Button("Auto Resolve", id="auto-resolve-btn")
                yield Button("Analyze", id="analyze-conflicts-btn")
            
            with Container(classes="conflict-content"):
                with Vertical(classes="conflicts-list"):
                    yield Label("Conflicted Files", classes="subsection-header")
                    yield ListView(id="conflicts-list")
                
                with Vertical(classes="conflict-details"):
                    yield Label("Conflict Analysis", classes="subsection-header")
                    yield Static("", id="conflict-analysis", classes="conflict-analysis-panel")
                    
                    yield Label("Resolution Actions", classes="subsection-header")
                    yield Container(
                        Button("Take Ours", id="take-ours-btn"),
                        Button("Take Theirs", id="take-theirs-btn"),
                        Button("Manual Edit", id="manual-edit-btn"),
                        classes="resolution-actions"
                    )
    
    @on(Button.Pressed, "#scan-conflicts-btn")
    async def scan_conflicts(self) -> None:
        """Scan for merge conflicts."""
        try:
            # Check for unmerged files
            if self.git_manager.repo:
                unmerged = [item[0] for item in self.git_manager.repo.index.unmerged_blobs()]
                
                self.current_conflicts = unmerged
                await self._update_conflicts_list()
                
                if unmerged:
                    self.notify(f"Found {len(unmerged)} conflicts", severity="warning")
                else:
                    self.notify("No conflicts found", severity="information")
            
        except Exception as e:
            logger.error(f"Conflict scan failed: {e}")
            self.notify(f"Conflict scan failed: {e}", severity="error")
    
    @on(Button.Pressed, "#analyze-conflicts-btn")
    async def analyze_conflicts(self) -> None:
        """Analyze current conflicts with AI."""
        if not self.current_conflicts:
            self.notify("No conflicts to analyze", severity="warning")
            return
        
        try:
            self.notify("Analyzing conflicts with AI...", severity="information")
            
            analysis = await self.git_manager.conflict_resolver.analyze_conflicts(
                self.git_manager.repo, self.current_conflicts
            )
            
            await self._display_conflict_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Conflict analysis failed: {e}")
            self.notify(f"Conflict analysis failed: {e}", severity="error")
    
    @on(Button.Pressed, "#auto-resolve-btn")
    async def auto_resolve_conflicts(self) -> None:
        """Attempt automatic conflict resolution."""
        if not self.current_conflicts:
            self.notify("No conflicts to resolve", severity="warning")
            return
        
        try:
            self.notify("Attempting automatic resolution...", severity="information")
            
            # First analyze conflicts
            analysis = await self.git_manager.conflict_resolver.analyze_conflicts(
                self.git_manager.repo, self.current_conflicts
            )
            
            # Then attempt auto-resolution
            resolution = await self.git_manager.conflict_resolver.auto_resolve_conflicts(
                self.git_manager.repo, analysis
            )
            
            resolved_count = len(resolution['resolved_files'])
            failed_count = len(resolution['failed_files'])
            manual_count = len(resolution['manual_review_files'])
            
            message = f"Resolved: {resolved_count}, Failed: {failed_count}, Manual: {manual_count}"
            self.notify(message, severity="information")
            
            # Refresh conflicts list
            await self.scan_conflicts()
            
        except Exception as e:
            logger.error(f"Auto-resolution failed: {e}")
            self.notify(f"Auto-resolution failed: {e}", severity="error")
    
    async def _update_conflicts_list(self) -> None:
        """Update the conflicts list display."""
        conflicts_list = self.query_one("#conflicts-list", ListView)
        conflicts_list.clear()
        
        for file_path in self.current_conflicts:
            # Truncate long paths
            display_path = file_path
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            
            conflicts_list.append(ListItem(Label(display_path)))
    
    async def _display_conflict_analysis(self, analysis: Dict[str, Any]) -> None:
        """Display conflict analysis results."""
        analysis_panel = self.query_one("#conflict-analysis", Static)
        
        total = analysis['total_conflicts']
        auto_resolvable = analysis['auto_resolvable']
        complex_conflicts = analysis['complex_conflicts']
        strategy = analysis['resolution_strategy']
        
        analysis_html = f"""
        <div class="conflict-analysis">
            <h3>Conflict Analysis Results</h3>
            <p><strong>Total Conflicts:</strong> {total}</p>
            <p><strong>Auto-resolvable:</strong> {auto_resolvable}</p>
            <p><strong>Complex Conflicts:</strong> {complex_conflicts}</p>
            <p><strong>Recommended Strategy:</strong> {strategy.replace('_', ' ').title()}</p>
            
            <h4>File Analysis</h4>
        """
        
        for file_analysis in analysis.get('file_analyses', [])[:5]:  # Show first 5
            file_path = file_analysis['file_path']
            conflict_count = file_analysis['conflict_count']
            complexity = file_analysis['complexity_score']
            resolvable = file_analysis['auto_resolvable']
            
            analysis_html += f"""
            <div class="file-analysis">
                <p><strong>{file_path}</strong></p>
                <p>Conflicts: {conflict_count}, Complexity: {complexity}, Auto-resolvable: {'Yes' if resolvable else 'No'}</p>
            </div>
            """
        
        analysis_html += "</div>"
        analysis_panel.update(analysis_html)


class GitWorkflowWidget(Static):
    """Main Git workflow widget combining all Git features."""
    
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "create_commit", "Smart Commit"),
        Binding("p", "create_pr", "Create PR"),
        Binding("m", "merge", "Smart Merge"),
        Binding("q", "quit", "Quit"),
    ]
    
    def __init__(self, repository_path: Optional[Path] = None, **kwargs):
        """Initialize main Git workflow widget."""
        super().__init__(**kwargs)
        self.git_manager = EnhancedGitManager(repository_path)
        self.repository_path = repository_path or Path.cwd()
    
    def compose(self) -> ComposeResult:
        """Compose the complete Git workflow interface."""
        with Container(classes="git-workflow-main"):
            yield Header(show_clock=True)
            
            with TabbedContent(id="main-tabs"):
                with TabPane("Status", id="status-tab"):
                    yield GitStatusWidget(self.git_manager, classes="git-status-widget")
                
                with TabPane("Branches", id="branches-tab"):
                    yield BranchTreeWidget(self.git_manager, classes="branch-tree-widget")
                
                with TabPane("Pull Requests", id="prs-tab"):
                    yield PullRequestDashboard(self.git_manager, classes="pr-dashboard-widget")
                
                with TabPane("Code Review", id="review-tab"):
                    yield CodeReviewWidget(self.git_manager, classes="code-review-widget")
                
                with TabPane("Conflicts", id="conflicts-tab"):
                    yield ConflictResolutionWidget(self.git_manager, classes="conflict-resolution-widget")
            
            yield Footer()
    
    def action_refresh(self) -> None:
        """Refresh current tab content."""
        # Get current tab and refresh its content
        self.notify("Refreshing Git data...", severity="information")
    
    async def action_create_commit(self) -> None:
        """Create a smart commit with AI-generated message."""
        try:
            self.notify("Creating smart commit...", severity="information")
            
            result = await self.git_manager.generate_smart_commit(
                auto_add=True,
                conventional_commits=True
            )
            
            if result.is_success:
                self.notify(f"Commit created: {result.commit_hash[:8]}", severity="information")
            else:
                self.notify(f"Commit failed: {result.message}", severity="error")
                
        except Exception as e:
            logger.error(f"Smart commit failed: {e}")
            self.notify(f"Smart commit failed: {e}", severity="error")
    
    async def action_create_pr(self) -> None:
        """Create a pull request."""
        try:
            self.notify("Creating pull request...", severity="information")
            
            status = await self.git_manager.get_repository_status()
            current_branch = status.get('current_branch')
            
            if not current_branch or current_branch in ['main', 'master']:
                self.notify("Switch to a feature branch first", severity="warning")
                return
            
            pr = await self.git_manager.create_pull_request(
                head_branch=current_branch,
                base_branch="main"
            )
            
            self.notify(f"PR #{pr.number} created successfully", severity="information")
            
        except Exception as e:
            logger.error(f"PR creation failed: {e}")
            self.notify(f"PR creation failed: {e}", severity="error")
    
    async def action_merge(self) -> None:
        """Perform smart merge with conflict resolution."""
        try:
            # This would need branch selection UI
            self.notify("Smart merge feature - select branch in UI", severity="information")
            
        except Exception as e:
            logger.error(f"Smart merge failed: {e}")
            self.notify(f"Smart merge failed: {e}", severity="error")


# CSS styles for the Git workflow widgets
GIT_WORKFLOW_CSS = """
/* Git Status Widget */
.git-status-container {
    border: solid $primary;
    height: 100%;
    padding: 1;
}

.git-status-grid {
    grid-size: 2 1;
    grid-gutter: 2 1;
    height: 100%;
}

.status-info {
    border: solid $secondary;
    padding: 1;
    height: 100%;
}

.file-status {
    border: solid $secondary;
    padding: 1;
    height: 100%;
}

.info-label {
    text-style: bold;
    color: $text;
}

.info-value {
    color: $text-muted;
    margin-bottom: 1;
}

.status-dirty {
    color: $warning;
}

.file-list {
    height: 8;
    border: solid $surface;
}

/* Branch Tree Widget */
.branch-tree-container {
    border: solid $primary;
    height: 100%;
    padding: 1;
}

.branch-controls {
    height: 3;
    margin-bottom: 1;
}

.branch-tree-content {
    layout: horizontal;
    height: 1fr;
}

#branch-tree {
    width: 2fr;
    border: solid $secondary;
    margin-right: 1;
}

.branch-details {
    width: 1fr;
    border: solid $secondary;
    padding: 1;
}

.branch-info-panel {
    height: 100%;
    padding: 1;
}

/* PR Dashboard */
.pr-dashboard-container {
    border: solid $primary;
    height: 100%;
    padding: 1;
}

.pr-controls {
    height: 3;
    margin-bottom: 1;
}

.pr-content {
    layout: horizontal;
    height: 1fr;
}

.pr-table {
    width: 3fr;
    border: solid $secondary;
    margin-right: 1;
}

.pr-details {
    width: 1fr;
    border: solid $secondary;
    padding: 1;
}

.pr-details-content {
    height: 100%;
    overflow-y: auto;
}

/* Code Review Widget */
.code-review-container {
    border: solid $primary;
    height: 100%;
    padding: 1;
}

.review-controls {
    height: 3;
    margin-bottom: 1;
}

.pr-input {
    width: 15;
    margin-right: 1;
}

.review-content {
    height: 1fr;
}

.review-summary {
    padding: 1;
    border: solid $surface;
    background: $surface;
}

.issues-list,
.comments-list,
.suggestions-list {
    height: 100%;
    border: solid $surface;
}

/* Conflict Resolution Widget */
.conflict-resolution-container {
    border: solid $primary;
    height: 100%;
    padding: 1;
}

.conflict-controls {
    height: 3;
    margin-bottom: 1;
}

.conflict-content {
    layout: horizontal;
    height: 1fr;
}

.conflicts-list {
    width: 1fr;
    border: solid $secondary;
    margin-right: 1;
    padding: 1;
}

.conflict-details {
    width: 1fr;
    border: solid $secondary;
    padding: 1;
}

.conflict-analysis-panel {
    height: 20;
    margin-bottom: 1;
    overflow-y: auto;
}

.resolution-actions {
    layout: horizontal;
}

/* Main Widget */
.git-workflow-main {
    height: 100vh;
}

.section-header {
    text-style: bold;
    color: $accent;
    text-align: center;
    margin-bottom: 1;
}

.subsection-header {
    text-style: bold;
    color: $text;
    margin-bottom: 1;
}
"""


# Export main widget
__all__ = [
    'GitWorkflowWidget',
    'GitStatusWidget', 
    'BranchTreeWidget',
    'PullRequestDashboard',
    'CodeReviewWidget',
    'ConflictResolutionWidget',
    'GIT_WORKFLOW_CSS'
]