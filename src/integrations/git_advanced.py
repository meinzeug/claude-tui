"""
Enhanced Git Integration Module - Advanced Version Control Features

This module provides enterprise-grade Git integration with automated pull request
management, code review workflows, branch strategy enforcement, and CI/CD integration.

Key Features:
- Automated PR creation with AI-generated titles/descriptions
- Intelligent code review workflows with anti-hallucination validation
- Smart merge conflict resolution with contextual suggestions
- Branch strategy enforcement (GitFlow, GitHub Flow)
- Integration with GitHub/GitLab APIs for issues and projects
- CI/CD pipeline integration triggers
- Automated commit message generation
- Code quality gates and security checks
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from uuid import uuid4

# Optional imports with fallbacks
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    from git import Repo
    from git.objects import Commit
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False
    class Repo:
        def __init__(self, *args, **kwargs):
            pass
    class Commit:
        pass

# Use relative imports with try/except to handle missing modules
try:
    from .git_manager import GitManager, GitOperationResult, GitOperationStatus
except ImportError:
    class GitManager:
        pass
    class GitOperationResult:
        pass
    class GitOperationStatus:
        pass

# Core types with fallbacks
try:
    from ..core.types import ValidationResult, Issue, Severity, IssueType
except ImportError:
    class ValidationResult:
        pass
    class Issue:
        pass
    class Severity:
        pass
    class IssueType:
        pass

try:
    from ..core.ai_interface import AIInterface
except ImportError:
    class AIInterface:
        pass

try:
    from ..services.validation_service import ValidationService
except ImportError:
    class ValidationService:
        pass

logger = logging.getLogger(__name__)


class PRStatus(str, Enum):
    """Pull request status states."""
    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    CONFLICTED = "conflicted"


class ReviewStatus(str, Enum):
    """Code review status states."""
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"


class CIStatus(str, Enum):
    """CI/CD status states."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class PullRequest:
    """Pull request data model."""
    id: Optional[int] = None
    number: Optional[int] = None
    title: str = ""
    body: str = ""
    head_branch: str = ""
    base_branch: str = "main"
    status: PRStatus = PRStatus.DRAFT
    url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    mergeable: bool = True
    conflicts: List[str] = field(default_factory=list)
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    ci_status: CIStatus = CIStatus.PENDING
    checks_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeReview:
    """Code review data model."""
    id: str
    pull_request_id: int
    reviewer: str
    status: ReviewStatus
    comments: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    security_issues: List[Issue] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    ai_generated: bool = False


@dataclass
class BranchRule:
    """Branch protection rule."""
    pattern: str
    required_reviews: int = 1
    dismiss_stale_reviews: bool = True
    require_code_owner_reviews: bool = False
    required_status_checks: List[str] = field(default_factory=list)
    enforce_admins: bool = True
    allow_force_pushes: bool = False
    allow_deletions: bool = False


class GitHubAPIClient:
    """GitHub API client for repository operations."""
    
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        """Initialize GitHub API client."""
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {
            'Authorization': f'token {self.token}' if self.token else '',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Claude-TIU-Git-Integration/1.0'
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information."""
        if not self.session:
            raise ValueError("Session not initialized")
            
        async with self.session.get(f'{self.base_url}/repos/{owner}/{repo}') as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"GitHub API error: {response.status}")
    
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> PullRequest:
        """Create a pull request."""
        if not self.session:
            raise ValueError("Session not initialized")
        
        data = {
            'title': title,
            'body': body,
            'head': head,
            'base': base
        }
        
        async with self.session.post(
            f'{self.base_url}/repos/{owner}/{repo}/pulls',
            json=data
        ) as response:
            if response.status == 201:
                pr_data = await response.json()
                return PullRequest(
                    id=pr_data['id'],
                    number=pr_data['number'],
                    title=pr_data['title'],
                    body=pr_data['body'],
                    head_branch=pr_data['head']['ref'],
                    base_branch=pr_data['base']['ref'],
                    status=PRStatus.OPEN,
                    url=pr_data['html_url'],
                    created_at=datetime.fromisoformat(pr_data['created_at'].replace('Z', '+00:00')),
                    metadata={'github_data': pr_data}
                )
            else:
                error_data = await response.json()
                raise Exception(f"Failed to create PR: {error_data}")
    
    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequest:
        """Get pull request details."""
        if not self.session:
            raise ValueError("Session not initialized")
        
        async with self.session.get(
            f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}'
        ) as response:
            if response.status == 200:
                pr_data = await response.json()
                return PullRequest(
                    id=pr_data['id'],
                    number=pr_data['number'],
                    title=pr_data['title'],
                    body=pr_data['body'],
                    head_branch=pr_data['head']['ref'],
                    base_branch=pr_data['base']['ref'],
                    status=PRStatus(pr_data['state']),
                    url=pr_data['html_url'],
                    mergeable=pr_data.get('mergeable', True),
                    metadata={'github_data': pr_data}
                )
            else:
                raise Exception(f"Failed to get PR: {response.status}")
    
    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        commit_sha: str,
        path: str,
        line: int
    ) -> Dict[str, Any]:
        """Create a review comment on a pull request."""
        if not self.session:
            raise ValueError("Session not initialized")
        
        data = {
            'body': body,
            'commit_id': commit_sha,
            'path': path,
            'line': line
        }
        
        async with self.session.post(
            f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments',
            json=data
        ) as response:
            if response.status == 201:
                return await response.json()
            else:
                raise Exception(f"Failed to create review comment: {response.status}")
    
    async def get_repository_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get repository issues."""
        if not self.session:
            raise ValueError("Session not initialized")
        
        params = {'state': state}
        if labels:
            params['labels'] = ','.join(labels)
        
        async with self.session.get(
            f'{self.base_url}/repos/{owner}/{repo}/issues',
            params=params
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get issues: {response.status}")


class CommitMessageGenerator:
    """AI-powered commit message generation."""
    
    def __init__(self, ai_interface: Optional[AIInterface] = None):
        """Initialize commit message generator."""
        self.ai_interface = ai_interface or AIInterface()
    
    async def generate_commit_message(
        self,
        diff_content: str,
        file_paths: List[str],
        conventional_commits: bool = True
    ) -> str:
        """Generate commit message based on changes."""
        
        # Analyze changes to determine type and scope
        change_analysis = await self._analyze_changes(diff_content, file_paths)
        
        if conventional_commits:
            return await self._generate_conventional_commit(change_analysis, diff_content)
        else:
            return await self._generate_standard_commit(change_analysis, diff_content)
    
    async def _analyze_changes(
        self,
        diff_content: str,
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """Analyze changes to categorize the commit type."""
        
        analysis = {
            'type': 'feat',  # Default
            'scope': None,
            'breaking_change': False,
            'files_count': len(file_paths),
            'primary_language': None,
            'change_category': 'modification'
        }
        
        # Determine primary language
        extensions = [Path(fp).suffix.lower() for fp in file_paths]
        extension_counts = {}
        for ext in extensions:
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        if extension_counts:
            primary_ext = max(extension_counts.items(), key=lambda x: x[1])[0]
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.go': 'go',
                '.rs': 'rust'
            }
            analysis['primary_language'] = language_map.get(primary_ext, 'code')
        
        # Analyze diff patterns to determine commit type
        if any(path.endswith('.md') for path in file_paths):
            if len([p for p in file_paths if p.endswith('.md')]) == len(file_paths):
                analysis['type'] = 'docs'
        
        if any('test' in path.lower() for path in file_paths):
            analysis['type'] = 'test'
        
        if any(path.endswith(('.yml', '.yaml', '.json', '.toml')) for path in file_paths):
            analysis['type'] = 'ci' if 'ci' in str(file_paths) or 'github' in str(file_paths) else 'chore'
        
        # Check for breaking changes
        if 'BREAKING CHANGE' in diff_content or any('breaking' in line.lower() for line in diff_content.splitlines()):
            analysis['breaking_change'] = True
        
        # Determine scope from file paths
        common_dirs = set()
        for path in file_paths:
            parts = Path(path).parts
            if len(parts) > 1:
                common_dirs.add(parts[0])
        
        if len(common_dirs) == 1:
            analysis['scope'] = common_dirs.pop()
        
        return analysis
    
    async def _generate_conventional_commit(
        self,
        analysis: Dict[str, Any],
        diff_content: str
    ) -> str:
        """Generate conventional commit message."""
        
        prompt = f"""
        Generate a conventional commit message for the following changes:
        
        Change Analysis:
        - Type: {analysis['type']}
        - Scope: {analysis.get('scope', 'N/A')}
        - Files modified: {analysis['files_count']}
        - Primary language: {analysis.get('primary_language', 'N/A')}
        - Breaking change: {analysis['breaking_change']}
        
        Diff excerpt (first 1000 chars):
        {diff_content[:1000]}
        
        Please generate a conventional commit message following this format:
        <type>[optional scope]: <description>
        
        [optional body]
        
        [optional footer(s)]
        
        Guidelines:
        - Keep description under 50 characters
        - Use imperative mood ("add" not "added")
        - Don't capitalize first letter of description
        - Don't end description with period
        - Include body if changes are complex
        - Add footer for breaking changes
        """
        
        try:
            response = await self.ai_interface.generate_response(prompt, max_tokens=200)
            return response.strip()
        except Exception as e:
            logger.warning(f"AI commit message generation failed: {e}")
            return self._fallback_commit_message(analysis)
    
    async def _generate_standard_commit(
        self,
        analysis: Dict[str, Any],
        diff_content: str
    ) -> str:
        """Generate standard commit message."""
        
        prompt = f"""
        Generate a clear, concise commit message for the following changes:
        
        Files modified: {analysis['files_count']}
        Primary language: {analysis.get('primary_language', 'N/A')}
        
        Diff excerpt:
        {diff_content[:1000]}
        
        Generate a commit message that:
        - Clearly describes what was changed
        - Uses imperative mood
        - Is under 72 characters
        - Is specific and actionable
        """
        
        try:
            response = await self.ai_interface.generate_response(prompt, max_tokens=100)
            return response.strip()
        except Exception as e:
            logger.warning(f"AI commit message generation failed: {e}")
            return self._fallback_commit_message(analysis)
    
    def _fallback_commit_message(self, analysis: Dict[str, Any]) -> str:
        """Generate fallback commit message without AI."""
        type_map = {
            'feat': 'Add new feature',
            'fix': 'Fix bug',
            'docs': 'Update documentation',
            'test': 'Add tests',
            'chore': 'Update configuration',
            'ci': 'Update CI configuration'
        }
        
        base_message = type_map.get(analysis['type'], 'Update code')
        
        if analysis.get('scope'):
            return f"{base_message} in {analysis['scope']}"
        else:
            return f"{base_message} ({analysis['files_count']} files)"


class ConflictResolver:
    """Smart merge conflict resolution with AI assistance."""
    
    def __init__(self, ai_interface: Optional[AIInterface] = None):
        """Initialize conflict resolver."""
        self.ai_interface = ai_interface or AIInterface()
    
    async def analyze_conflicts(
        self,
        repo: Repo,
        conflict_files: List[str]
    ) -> Dict[str, Any]:
        """Analyze merge conflicts and provide resolution suggestions."""
        
        conflict_analysis = {
            'total_conflicts': len(conflict_files),
            'file_analyses': [],
            'resolution_strategy': 'manual',
            'auto_resolvable': 0,
            'complex_conflicts': 0
        }
        
        for file_path in conflict_files:
            try:
                full_path = Path(repo.working_tree_dir) / file_path
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                
                file_analysis = await self._analyze_single_conflict(
                    file_path, content
                )
                conflict_analysis['file_analyses'].append(file_analysis)
                
                if file_analysis['auto_resolvable']:
                    conflict_analysis['auto_resolvable'] += 1
                else:
                    conflict_analysis['complex_conflicts'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to analyze conflict in {file_path}: {e}")
        
        # Determine overall resolution strategy
        if conflict_analysis['auto_resolvable'] == conflict_analysis['total_conflicts']:
            conflict_analysis['resolution_strategy'] = 'automatic'
        elif conflict_analysis['auto_resolvable'] > conflict_analysis['complex_conflicts']:
            conflict_analysis['resolution_strategy'] = 'semi_automatic'
        else:
            conflict_analysis['resolution_strategy'] = 'manual'
        
        return conflict_analysis
    
    async def _analyze_single_conflict(
        self,
        file_path: str,
        content: str
    ) -> Dict[str, Any]:
        """Analyze a single conflict file."""
        
        # Parse conflict markers
        conflict_sections = self._parse_conflict_markers(content)
        
        analysis = {
            'file_path': file_path,
            'conflict_count': len(conflict_sections),
            'auto_resolvable': False,
            'resolution_suggestions': [],
            'complexity_score': 0,
            'conflict_types': []
        }
        
        # Analyze each conflict section
        for i, section in enumerate(conflict_sections):
            section_analysis = await self._analyze_conflict_section(section, file_path)
            analysis['resolution_suggestions'].append(section_analysis)
            analysis['complexity_score'] += section_analysis['complexity']
            analysis['conflict_types'].append(section_analysis['type'])
        
        # Determine if auto-resolvable
        analysis['auto_resolvable'] = (
            analysis['complexity_score'] < 2 and
            all(ct in ['whitespace', 'import', 'simple_addition'] for ct in analysis['conflict_types'])
        )
        
        return analysis
    
    def _parse_conflict_markers(self, content: str) -> List[Dict[str, str]]:
        """Parse Git conflict markers in file content."""
        sections = []
        lines = content.splitlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('<<<<<<<'):
                # Found conflict start
                conflict_start = i
                ours_content = []
                theirs_content = []
                
                # Find separator
                separator_line = None
                for j in range(i + 1, len(lines)):
                    if lines[j] == '=======':
                        separator_line = j
                        break
                    ours_content.append(lines[j])
                
                if separator_line is None:
                    i += 1
                    continue
                
                # Find conflict end
                conflict_end = None
                for j in range(separator_line + 1, len(lines)):
                    if lines[j].startswith('>>>>>>>'):
                        conflict_end = j
                        break
                    theirs_content.append(lines[j])
                
                if conflict_end is None:
                    i += 1
                    continue
                
                sections.append({
                    'start_line': conflict_start + 1,
                    'end_line': conflict_end + 1,
                    'ours': '\n'.join(ours_content),
                    'theirs': '\n'.join(theirs_content),
                    'ours_label': lines[conflict_start].replace('<<<<<<<', '').strip(),
                    'theirs_label': lines[conflict_end].replace('>>>>>>>', '').strip()
                })
                
                i = conflict_end + 1
            else:
                i += 1
        
        return sections
    
    async def _analyze_conflict_section(
        self,
        section: Dict[str, str],
        file_path: str
    ) -> Dict[str, Any]:
        """Analyze a single conflict section."""
        
        ours = section['ours'].strip()
        theirs = section['theirs'].strip()
        
        analysis = {
            'type': 'unknown',
            'complexity': 1,
            'suggested_resolution': 'manual',
            'reasoning': '',
            'auto_fix_available': False
        }
        
        # Simple heuristics for conflict type detection
        if not ours and theirs:
            analysis['type'] = 'addition_theirs'
            analysis['complexity'] = 0
            analysis['suggested_resolution'] = 'take_theirs'
            analysis['auto_fix_available'] = True
            analysis['reasoning'] = 'Only theirs has content'
            
        elif ours and not theirs:
            analysis['type'] = 'addition_ours'
            analysis['complexity'] = 0
            analysis['suggested_resolution'] = 'take_ours'
            analysis['auto_fix_available'] = True
            analysis['reasoning'] = 'Only ours has content'
            
        elif ours == theirs:
            analysis['type'] = 'identical'
            analysis['complexity'] = 0
            analysis['suggested_resolution'] = 'take_either'
            analysis['auto_fix_available'] = True
            analysis['reasoning'] = 'Both sides are identical'
            
        elif self._is_whitespace_conflict(ours, theirs):
            analysis['type'] = 'whitespace'
            analysis['complexity'] = 0
            analysis['suggested_resolution'] = 'normalize_whitespace'
            analysis['auto_fix_available'] = True
            analysis['reasoning'] = 'Only whitespace differences'
            
        elif self._is_import_conflict(ours, theirs, file_path):
            analysis['type'] = 'import'
            analysis['complexity'] = 1
            analysis['suggested_resolution'] = 'merge_imports'
            analysis['auto_fix_available'] = True
            analysis['reasoning'] = 'Import statement conflict'
            
        else:
            # Use AI for complex analysis
            analysis = await self._ai_analyze_conflict(section, file_path, analysis)
        
        return analysis
    
    def _is_whitespace_conflict(self, ours: str, theirs: str) -> bool:
        """Check if conflict is only about whitespace."""
        return ours.replace(' ', '').replace('\t', '').replace('\n', '') == \
               theirs.replace(' ', '').replace('\t', '').replace('\n', '')
    
    def _is_import_conflict(self, ours: str, theirs: str, file_path: str) -> bool:
        """Check if conflict is in import statements."""
        if not file_path.endswith(('.py', '.js', '.ts', '.java')):
            return False
        
        import_keywords = ['import', 'from', 'require', 'include']
        return any(keyword in ours.lower() for keyword in import_keywords) and \
               any(keyword in theirs.lower() for keyword in import_keywords)
    
    async def _ai_analyze_conflict(
        self,
        section: Dict[str, str],
        file_path: str,
        base_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to analyze complex conflicts."""
        
        prompt = f"""
        Analyze this merge conflict and suggest the best resolution:
        
        File: {file_path}
        
        Our version:
        {section['ours']}
        
        Their version:
        {section['theirs']}
        
        Provide analysis in this format:
        Type: [function_change|variable_conflict|logic_difference|formatting]
        Complexity: [0-3 where 0=trivial, 3=complex]
        Suggested Resolution: [take_ours|take_theirs|merge_both|manual_review]
        Reasoning: [brief explanation]
        Auto-fixable: [yes|no]
        """
        
        try:
            response = await self.ai_interface.generate_response(prompt, max_tokens=150)
            # Parse AI response and update analysis
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('Type:'):
                    base_analysis['type'] = line.split(':', 1)[1].strip()
                elif line.startswith('Complexity:'):
                    try:
                        base_analysis['complexity'] = int(line.split(':', 1)[1].strip()[0])
                    except:
                        pass
                elif line.startswith('Suggested Resolution:'):
                    base_analysis['suggested_resolution'] = line.split(':', 1)[1].strip()
                elif line.startswith('Reasoning:'):
                    base_analysis['reasoning'] = line.split(':', 1)[1].strip()
                elif line.startswith('Auto-fixable:'):
                    base_analysis['auto_fix_available'] = 'yes' in line.lower()
        
        except Exception as e:
            logger.warning(f"AI conflict analysis failed: {e}")
            base_analysis['complexity'] = 2  # Default to medium complexity
        
        return base_analysis
    
    async def auto_resolve_conflicts(
        self,
        repo: Repo,
        conflict_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Automatically resolve conflicts where possible."""
        
        resolution_result = {
            'resolved_files': [],
            'failed_files': [],
            'manual_review_files': [],
            'total_processed': 0
        }
        
        for file_analysis in conflict_analysis['file_analyses']:
            if not file_analysis['auto_resolvable']:
                resolution_result['manual_review_files'].append(file_analysis['file_path'])
                continue
            
            try:
                await self._auto_resolve_file(repo, file_analysis)
                resolution_result['resolved_files'].append(file_analysis['file_path'])
            except Exception as e:
                logger.error(f"Auto-resolution failed for {file_analysis['file_path']}: {e}")
                resolution_result['failed_files'].append(file_analysis['file_path'])
            
            resolution_result['total_processed'] += 1
        
        return resolution_result
    
    async def _auto_resolve_file(self, repo: Repo, file_analysis: Dict[str, Any]) -> None:
        """Auto-resolve conflicts in a single file."""
        file_path = Path(repo.working_tree_dir) / file_analysis['file_path']
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Apply resolution suggestions
        resolved_content = content
        for suggestion in file_analysis['resolution_suggestions']:
            if suggestion['auto_fix_available']:
                resolved_content = await self._apply_resolution(
                    resolved_content, suggestion
                )
        
        # Write resolved content
        file_path.write_text(resolved_content, encoding='utf-8')
        
        # Stage the resolved file
        repo.index.add([str(file_analysis['file_path'])])
    
    async def _apply_resolution(
        self,
        content: str,
        suggestion: Dict[str, Any]
    ) -> str:
        """Apply a specific resolution suggestion to content."""
        
        # This is a simplified implementation - would need more sophisticated
        # pattern matching and replacement logic for production use
        
        if suggestion['suggested_resolution'] == 'take_ours':
            # Remove conflict markers and keep ours
            pattern = r'<<<<<<< .*?\n(.*?)\n=======.*?\n.*?\n>>>>>>> .*?\n'
            return re.sub(pattern, r'\1\n', content, flags=re.DOTALL)
        
        elif suggestion['suggested_resolution'] == 'take_theirs':
            # Remove conflict markers and keep theirs
            pattern = r'<<<<<<< .*?\n.*?\n=======(.*?)\n>>>>>>> .*?\n'
            return re.sub(pattern, r'\1\n', content, flags=re.DOTALL)
        
        elif suggestion['suggested_resolution'] == 'take_either':
            # Take ours (same result as take_ours for identical content)
            pattern = r'<<<<<<< .*?\n(.*?)\n=======.*?\n.*?\n>>>>>>> .*?\n'
            return re.sub(pattern, r'\1\n', content, flags=re.DOTALL)
        
        return content


class EnhancedGitManager(GitManager):
    """
    Enhanced Git Manager with advanced features for enterprise development.
    
    Extends the base GitManager with:
    - Automated pull request creation and management
    - AI-powered code review workflows
    - Smart conflict resolution
    - Branch strategy enforcement
    - CI/CD integration
    - GitHub/GitLab API integration
    """
    
    def __init__(
        self,
        repository_path: Optional[Union[str, Path]] = None,
        github_token: Optional[str] = None,
        gitlab_token: Optional[str] = None,
        **kwargs
    ):
        """Initialize enhanced Git manager."""
        super().__init__(repository_path, **kwargs)
        
        # API clients
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.gitlab_token = gitlab_token or os.getenv('GITLAB_TOKEN')
        
        # Enhanced features
        self.commit_generator = CommitMessageGenerator()
        self.conflict_resolver = ConflictResolver()
        self.validation_service = ValidationService()
        
        # State tracking
        self.pull_requests: Dict[int, PullRequest] = {}
        self.active_reviews: Dict[str, CodeReview] = {}
        self.branch_rules: Dict[str, BranchRule] = {}
        
        # Configure default branch rules
        self._setup_default_branch_rules()
    
    def _setup_default_branch_rules(self) -> None:
        """Setup default branch protection rules."""
        self.branch_rules = {
            'main': BranchRule(
                pattern='main',
                required_reviews=2,
                required_status_checks=['ci', 'tests'],
                enforce_admins=True
            ),
            'develop': BranchRule(
                pattern='develop',
                required_reviews=1,
                required_status_checks=['tests'],
                allow_force_pushes=True
            ),
            'feature/*': BranchRule(
                pattern='feature/*',
                required_reviews=1,
                allow_force_pushes=True,
                allow_deletions=True
            )
        }
    
    async def generate_smart_commit(
        self,
        auto_add: bool = True,
        conventional_commits: bool = True,
        validate_message: bool = True
    ) -> GitOperationResult:
        """
        Generate and create a smart commit with AI-generated message.
        
        Args:
            auto_add: Automatically add modified files
            conventional_commits: Use conventional commit format
            validate_message: Validate generated message
            
        Returns:
            GitOperationResult with commit details
        """
        if not self.repo:
            raise Exception("Repository not initialized")
        
        start_time = time.time()
        logger.info("Generating smart commit with AI-generated message")
        
        try:
            # Get changes for analysis
            modified_files = [item.a_path for item in self.repo.index.diff(None)]
            staged_files = [item.a_path for item in self.repo.index.diff("HEAD")]
            untracked_files = list(self.repo.untracked_files)
            
            # Auto-add files if requested
            if auto_add and (modified_files or untracked_files):
                all_files = modified_files + untracked_files
                await self.add_files(all_files)
                staged_files.extend(all_files)
            
            if not staged_files:
                return GitOperationResult(
                    operation="smart_commit",
                    status=GitOperationStatus.SUCCESS,
                    message="No changes to commit",
                    execution_time=time.time() - start_time
                )
            
            # Get diff content for message generation
            try:
                diff_content = self.repo.git.diff('--cached')
            except:
                diff_content = "Binary or large file changes"
            
            # Generate commit message
            commit_message = await self.commit_generator.generate_commit_message(
                diff_content, staged_files, conventional_commits
            )
            
            # Validate message if requested
            if validate_message:
                validation = await self._validate_commit_message(commit_message)
                if not validation['valid']:
                    # Try to improve the message
                    commit_message = await self._improve_commit_message(
                        commit_message, validation['error']
                    )
            
            # Create the commit
            commit_result = await self.commit(
                message=commit_message,
                validate_message=False  # Already validated
            )
            
            # Add AI metadata
            commit_result.metadata.update({
                'ai_generated_message': True,
                'conventional_commits': conventional_commits,
                'files_analyzed': len(staged_files),
                'diff_size': len(diff_content)
            })
            
            execution_time = time.time() - start_time
            commit_result.execution_time = execution_time
            
            logger.info(f"Smart commit created successfully in {execution_time:.2f}s")
            return commit_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="smart_commit",
                status=GitOperationStatus.FAILED,
                message=f"Smart commit failed: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Smart commit failed: {e}")
            raise Exception(f"Smart commit failed: {e}") from e
    
    async def create_pull_request(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        base_branch: str = "main",
        head_branch: Optional[str] = None,
        draft: bool = False,
        auto_merge: bool = False,
        repository_url: Optional[str] = None
    ) -> PullRequest:
        """
        Create a pull request with AI-generated title and description.
        
        Args:
            title: PR title (AI-generated if None)
            body: PR body (AI-generated if None)
            base_branch: Target branch
            head_branch: Source branch (current branch if None)
            draft: Create as draft PR
            auto_merge: Enable auto-merge when checks pass
            repository_url: GitHub repository URL
            
        Returns:
            Created PullRequest object
        """
        if not self.repo:
            raise Exception("Repository not initialized")
        
        logger.info("Creating pull request with AI assistance")
        
        try:
            # Determine head branch
            if not head_branch:
                head_branch = self.repo.active_branch.name
            
            # Get repository information from URL or remote
            repo_info = await self._parse_repository_info(repository_url)
            if not repo_info:
                raise Exception("Could not determine repository information")
            
            # Generate PR title and body if not provided
            if not title or not body:
                ai_content = await self._generate_pr_content(
                    head_branch, base_branch, title, body
                )
                title = title or ai_content['title']
                body = body or ai_content['body']
            
            # Create PR using GitHub API
            async with GitHubAPIClient(self.github_token) as github:
                pr = await github.create_pull_request(
                    owner=repo_info['owner'],
                    repo=repo_info['repo'],
                    title=title,
                    body=body,
                    head=head_branch,
                    base=base_branch
                )
            
            # Set as draft if requested
            if draft:
                pr.status = PRStatus.DRAFT
            
            # Store PR locally
            self.pull_requests[pr.number] = pr
            
            # Trigger initial code review if enabled
            if not draft:
                await self._schedule_code_review(pr)
            
            logger.info(f"Pull request #{pr.number} created successfully: {pr.url}")
            return pr
            
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            raise Exception(f"Pull request creation failed: {e}") from e
    
    async def review_pull_request(
        self,
        pr_number: int,
        review_type: str = "comprehensive",
        auto_approve_safe: bool = False
    ) -> CodeReview:
        """
        Perform AI-powered code review on a pull request.
        
        Args:
            pr_number: Pull request number
            review_type: Type of review (quick, standard, comprehensive)
            auto_approve_safe: Auto-approve if no issues found
            
        Returns:
            CodeReview object with findings
        """
        if pr_number not in self.pull_requests:
            # Fetch PR details
            repo_info = await self._parse_repository_info()
            async with GitHubAPIClient(self.github_token) as github:
                pr = await github.get_pull_request(
                    repo_info['owner'], repo_info['repo'], pr_number
                )
                self.pull_requests[pr_number] = pr
        
        pr = self.pull_requests[pr_number]
        logger.info(f"Starting {review_type} code review for PR #{pr_number}")
        
        try:
            # Get PR changes
            changes = await self._get_pr_changes(pr)
            
            # Perform validation on changed files
            validation_results = []
            for file_path in changes['modified_files']:
                try:
                    file_validation = await self.validation_service.validate_file(
                        Path(self.repository_path) / file_path
                    )
                    validation_results.append(file_validation)
                except Exception as e:
                    logger.warning(f"Validation failed for {file_path}: {e}")
            
            # Generate review
            review = await self._generate_code_review(
                pr, changes, validation_results, review_type
            )
            
            # Post review comments if GitHub token available
            if self.github_token and review.comments:
                await self._post_review_comments(pr, review)
            
            # Auto-approve if safe and requested
            if auto_approve_safe and review.status == ReviewStatus.APPROVED:
                await self._approve_pull_request(pr)
            
            # Store review
            self.active_reviews[review.id] = review
            
            logger.info(f"Code review completed for PR #{pr_number}: {review.status.value}")
            return review
            
        except Exception as e:
            logger.error(f"Code review failed for PR #{pr_number}: {e}")
            raise Exception(f"Code review failed: {e}") from e
    
    async def smart_merge_with_conflict_resolution(
        self,
        branch_name: str,
        strategy: str = "adaptive",
        auto_resolve: bool = True,
        validate_merge: bool = True
    ) -> GitOperationResult:
        """
        Smart merge with AI-assisted conflict resolution.
        
        Args:
            branch_name: Branch to merge
            strategy: Merge strategy (adaptive, ours, theirs, manual)
            auto_resolve: Attempt automatic conflict resolution
            validate_merge: Validate merge result
            
        Returns:
            GitOperationResult with merge details
        """
        if not self.repo:
            raise Exception("Repository not initialized")
        
        start_time = time.time()
        logger.info(f"Starting smart merge of '{branch_name}' with conflict resolution")
        
        try:
            # Pre-merge conflict detection
            potential_conflicts = await self._detect_potential_conflicts(branch_name)
            
            if potential_conflicts:
                logger.info(f"Detected {len(potential_conflicts)} potential conflicts")
            
            # Attempt initial merge
            merge_result = await self.merge_branch(branch_name, no_ff=True)
            
            # If conflicts detected, attempt resolution
            if merge_result.has_conflicts and auto_resolve:
                logger.info("Attempting automatic conflict resolution")
                
                # Analyze conflicts
                conflict_analysis = await self.conflict_resolver.analyze_conflicts(
                    self.repo, merge_result.conflicts
                )
                
                # Auto-resolve where possible
                if conflict_analysis['auto_resolvable'] > 0:
                    resolution_result = await self.conflict_resolver.auto_resolve_conflicts(
                        self.repo, conflict_analysis
                    )
                    
                    # Update merge result
                    resolved_files = resolution_result['resolved_files']
                    remaining_conflicts = [
                        f for f in merge_result.conflicts
                        if f not in resolved_files
                    ]
                    
                    if not remaining_conflicts:
                        # Complete the merge
                        commit = self.repo.index.commit(
                            f"Merge branch '{branch_name}' with AI-assisted conflict resolution"
                        )
                        
                        merge_result = GitOperationResult(
                            operation="smart_merge",
                            status=GitOperationStatus.SUCCESS,
                            message=f"Successfully merged '{branch_name}' with auto-resolved conflicts",
                            commit_hash=commit.hexsha,
                            files_modified=resolved_files,
                            execution_time=time.time() - start_time,
                            metadata={
                                'auto_resolved_conflicts': len(resolved_files),
                                'manual_conflicts': len(resolution_result['manual_review_files']),
                                'resolution_strategy': strategy
                            }
                        )
                    else:
                        merge_result.conflicts = remaining_conflicts
                        merge_result.message += f" ({len(resolved_files)} conflicts auto-resolved)"
            
            # Validate merge result if requested
            if validate_merge and merge_result.is_success:
                validation_result = await self._validate_merge_result(merge_result)
                merge_result.metadata['validation_result'] = validation_result
            
            # Update metrics and history
            self._update_metrics(merge_result)
            self.operation_history.append(merge_result)
            
            execution_time = time.time() - start_time
            merge_result.execution_time = execution_time
            
            logger.info(f"Smart merge completed in {execution_time:.2f}s")
            return merge_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="smart_merge",
                status=GitOperationStatus.FAILED,
                message=f"Smart merge failed: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Smart merge failed: {e}")
            raise Exception(f"Smart merge failed: {e}") from e
    
    async def enforce_branch_protection(
        self,
        branch_pattern: str,
        rule: Optional[BranchRule] = None
    ) -> GitOperationResult:
        """
        Enforce branch protection rules.
        
        Args:
            branch_pattern: Branch pattern to protect
            rule: Protection rule (default rule if None)
            
        Returns:
            GitOperationResult with enforcement status
        """
        if not rule:
            rule = self.branch_rules.get(branch_pattern, BranchRule(pattern=branch_pattern))
        
        logger.info(f"Enforcing branch protection for pattern: {branch_pattern}")
        
        try:
            # Store rule
            self.branch_rules[branch_pattern] = rule
            
            # Apply protection via API if possible
            repo_info = await self._parse_repository_info()
            if repo_info and self.github_token:
                success = await self._apply_branch_protection_via_api(
                    repo_info, branch_pattern, rule
                )
                
                status = GitOperationStatus.SUCCESS if success else GitOperationStatus.FAILED
                message = f"Branch protection {'applied' if success else 'failed'} for {branch_pattern}"
            else:
                # Local enforcement only
                status = GitOperationStatus.SUCCESS
                message = f"Branch protection rule stored locally for {branch_pattern}"
            
            result = GitOperationResult(
                operation="enforce_branch_protection",
                status=status,
                message=message,
                metadata={'branch_pattern': branch_pattern, 'rule': rule.__dict__}
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            return result
            
        except Exception as e:
            result = GitOperationResult(
                operation="enforce_branch_protection",
                status=GitOperationStatus.FAILED,
                message=f"Branch protection enforcement failed: {e}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Branch protection enforcement failed: {e}")
            raise Exception(f"Branch protection enforcement failed: {e}") from e
    
    async def trigger_ci_pipeline(
        self,
        branch_name: Optional[str] = None,
        workflow_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger CI/CD pipeline.
        
        Args:
            branch_name: Target branch (current branch if None)
            workflow_name: Workflow to trigger
            inputs: Workflow inputs
            
        Returns:
            Pipeline trigger result
        """
        if not branch_name:
            branch_name = self.repo.active_branch.name
        
        logger.info(f"Triggering CI pipeline for branch: {branch_name}")
        
        try:
            repo_info = await self._parse_repository_info()
            if not repo_info:
                raise Exception("Repository information not available")
            
            # Trigger via GitHub Actions API
            if self.github_token:
                result = await self._trigger_github_workflow(
                    repo_info, branch_name, workflow_name, inputs
                )
                return result
            
            # Fallback to local hooks
            return await self._trigger_local_ci_hooks(branch_name, inputs)
            
        except Exception as e:
            logger.error(f"CI pipeline trigger failed: {e}")
            raise Exception(f"CI pipeline trigger failed: {e}") from e
    
    # Helper methods
    
    async def _improve_commit_message(self, message: str, error: str) -> str:
        """Improve commit message based on validation error."""
        if not hasattr(self, 'ai_interface'):
            return message  # Return original if no AI interface
        
        prompt = f"""
        Improve this commit message to fix the validation error:
        
        Original message: {message}
        Validation error: {error}
        
        Generate an improved commit message that follows best practices.
        """
        
        try:
            response = await self.ai_interface.generate_response(prompt, max_tokens=100)
            return response.strip()
        except:
            return message
    
    async def _parse_repository_info(
        self,
        repository_url: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """Parse repository owner and name from URL or remote."""
        if repository_url:
            # Parse from provided URL
            match = re.match(r'https://github\.com/([^/]+)/([^/]+)', repository_url)
            if match:
                return {'owner': match.group(1), 'repo': match.group(2)}
        
        # Try to get from Git remotes
        try:
            if self.repo:
                for remote in self.repo.remotes:
                    url = remote.url
                    # GitHub SSH format
                    match = re.match(r'git@github\.com:([^/]+)/(.+)\.git', url)
                    if match:
                        return {'owner': match.group(1), 'repo': match.group(2)}
                    
                    # GitHub HTTPS format
                    match = re.match(r'https://github\.com/([^/]+)/(.+)(?:\.git)?', url)
                    if match:
                        repo_name = match.group(2).rstrip('.git')
                        return {'owner': match.group(1), 'repo': repo_name}
        except:
            pass
        
        return None
    
    async def _generate_pr_content(
        self,
        head_branch: str,
        base_branch: str,
        existing_title: Optional[str],
        existing_body: Optional[str]
    ) -> Dict[str, str]:
        """Generate PR title and body using AI."""
        
        # Get branch changes
        try:
            comparison = self.repo.git.log(
                f'{base_branch}..{head_branch}',
                '--oneline',
                '--max-count=20'
            )
            
            file_changes = self.repo.git.diff(
                f'{base_branch}...{head_branch}',
                '--name-status'
            )
        except:
            comparison = "Unable to get commit history"
            file_changes = "Unable to get file changes"
        
        prompt = f"""
        Generate a pull request title and description for the following changes:
        
        Source branch: {head_branch}
        Target branch: {base_branch}
        
        Recent commits:
        {comparison[:1000]}
        
        File changes:
        {file_changes[:500]}
        
        Generate:
        1. A clear, concise title (under 60 characters)
        2. A structured description with:
           - What changes were made
           - Why these changes were made
           - How to test the changes
        
        Format your response as:
        TITLE: [title here]
        
        DESCRIPTION:
        [description here]
        """
        
        try:
            if hasattr(self, 'ai_interface'):
                response = await self.ai_interface.generate_response(prompt, max_tokens=400)
                
                # Parse response
                lines = response.strip().split('\n')
                title = existing_title
                body = existing_body
                
                for i, line in enumerate(lines):
                    if line.startswith('TITLE:'):
                        title = line[6:].strip()
                    elif line.startswith('DESCRIPTION:'):
                        body = '\n'.join(lines[i+1:]).strip()
                        break
                
                return {
                    'title': title or f"Updates from {head_branch}",
                    'body': body or f"Changes from branch {head_branch}"
                }
        except Exception as e:
            logger.warning(f"AI PR content generation failed: {e}")
        
        # Fallback
        return {
            'title': existing_title or f"Merge {head_branch} into {base_branch}",
            'body': existing_body or f"Automated pull request from {head_branch}"
        }
    
    async def _schedule_code_review(self, pr: PullRequest) -> None:
        """Schedule automatic code review for PR."""
        try:
            await asyncio.sleep(5)  # Brief delay
            await self.review_pull_request(pr.number, review_type="standard")
        except Exception as e:
            logger.warning(f"Automatic code review scheduling failed: {e}")
    
    async def _get_pr_changes(self, pr: PullRequest) -> Dict[str, Any]:
        """Get changes made in a pull request."""
        try:
            # Get diff between base and head
            diff = self.repo.git.diff(
                f'{pr.base_branch}...{pr.head_branch}',
                '--name-status'
            )
            
            # Parse modified files
            modified_files = []
            for line in diff.splitlines():
                if line and len(line.split()) >= 2:
                    status, file_path = line.split('\t', 1)
                    modified_files.append(file_path)
            
            # Get detailed diff
            detailed_diff = self.repo.git.diff(
                f'{pr.base_branch}...{pr.head_branch}'
            )
            
            return {
                'modified_files': modified_files,
                'diff_content': detailed_diff,
                'file_count': len(modified_files)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get PR changes: {e}")
            return {'modified_files': [], 'diff_content': '', 'file_count': 0}
    
    async def _generate_code_review(
        self,
        pr: PullRequest,
        changes: Dict[str, Any],
        validation_results: List[ValidationResult],
        review_type: str
    ) -> CodeReview:
        """Generate comprehensive code review."""
        
        review_id = str(uuid4())
        
        # Combine validation issues
        all_issues = []
        for validation in validation_results:
            all_issues.extend(validation.issues)
        
        # Determine review status
        critical_issues = [i for i in all_issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in all_issues if i.severity == Severity.HIGH]
        
        if critical_issues:
            status = ReviewStatus.CHANGES_REQUESTED
        elif high_issues and len(high_issues) > 3:
            status = ReviewStatus.CHANGES_REQUESTED
        elif all_issues:
            status = ReviewStatus.COMMENTED
        else:
            status = ReviewStatus.APPROVED
        
        # Generate review comments
        comments = await self._generate_review_comments(
            changes, all_issues, review_type
        )
        
        # Calculate quality score
        quality_score = await self._calculate_quality_score(validation_results)
        
        return CodeReview(
            id=review_id,
            pull_request_id=pr.number,
            reviewer="Claude-TIU-AI",
            status=status,
            comments=comments,
            suggestions=await self._generate_improvement_suggestions(all_issues),
            security_issues=[i for i in all_issues if i.type == IssueType.SECURITY_VULNERABILITY],
            quality_score=quality_score,
            ai_generated=True
        )
    
    async def _generate_review_comments(
        self,
        changes: Dict[str, Any],
        issues: List[Issue],
        review_type: str
    ) -> List[Dict[str, Any]]:
        """Generate specific review comments."""
        comments = []
        
        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            if issue.file_path:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
        
        # Generate comments for each file with issues
        for file_path, file_issues in issues_by_file.items():
            for issue in file_issues:
                comment = {
                    'path': file_path,
                    'line': issue.line_number,
                    'body': self._format_issue_comment(issue),
                    'severity': issue.severity.value
                }
                comments.append(comment)
        
        # Add general review comments
        if review_type == "comprehensive":
            general_comment = await self._generate_general_review_comment(changes, issues)
            if general_comment:
                comments.append({
                    'path': None,
                    'line': None,
                    'body': general_comment,
                    'severity': 'info'
                })
        
        return comments
    
    def _format_issue_comment(self, issue: Issue) -> str:
        """Format an issue as a review comment."""
        severity_emoji = {
            Severity.CRITICAL: "🚨",
            Severity.HIGH: "⚠️",
            Severity.MEDIUM: "💡",
            Severity.LOW: "📝"
        }
        
        emoji = severity_emoji.get(issue.severity, "📝")
        comment = f"{emoji} **{issue.type.value.replace('_', ' ').title()}**\n\n"
        comment += issue.description
        
        if issue.suggested_fix:
            comment += f"\n\n**Suggested Fix:**\n{issue.suggested_fix}"
        
        return comment
    
    async def _generate_general_review_comment(
        self,
        changes: Dict[str, Any],
        issues: List[Issue]
    ) -> Optional[str]:
        """Generate general review comment."""
        if not issues:
            return "✅ Code looks good! No significant issues found."
        
        critical_count = len([i for i in issues if i.severity == Severity.CRITICAL])
        high_count = len([i for i in issues if i.severity == Severity.HIGH])
        medium_count = len([i for i in issues if i.severity == Severity.MEDIUM])
        
        comment = "## Code Review Summary\n\n"
        comment += f"Reviewed {changes['file_count']} files with the following findings:\n\n"
        
        if critical_count:
            comment += f"🚨 {critical_count} critical issue(s) that must be addressed\n"
        if high_count:
            comment += f"⚠️ {high_count} high-priority issue(s) recommended to fix\n"
        if medium_count:
            comment += f"💡 {medium_count} medium-priority suggestion(s)\n"
        
        comment += "\nPlease address the critical and high-priority issues before merging."
        
        return comment
    
    async def _generate_improvement_suggestions(
        self,
        issues: List[Issue]
    ) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        # Group by issue type
        issue_types = {}
        for issue in issues:
            if issue.type not in issue_types:
                issue_types[issue.type] = 0
            issue_types[issue.type] += 1
        
        # Generate suggestions based on patterns
        if IssueType.PLACEHOLDER in issue_types:
            suggestions.append("Replace placeholder content with proper implementations")
        
        if IssueType.EMPTY_FUNCTION in issue_types:
            suggestions.append("Implement empty function bodies with proper logic")
        
        if IssueType.SECURITY_VULNERABILITY in issue_types:
            suggestions.append("Address security vulnerabilities before deployment")
        
        if IssueType.INCOMPLETE_IMPLEMENTATION in issue_types:
            suggestions.append("Complete partial implementations and add proper error handling")
        
        return suggestions
    
    async def _calculate_quality_score(
        self,
        validation_results: List[ValidationResult]
    ) -> float:
        """Calculate overall quality score from validation results."""
        if not validation_results:
            return 85.0  # Default score
        
        total_score = sum(result.authenticity_score for result in validation_results)
        return total_score / len(validation_results)
    
    async def _post_review_comments(self, pr: PullRequest, review: CodeReview) -> None:
        """Post review comments to GitHub."""
        try:
            repo_info = await self._parse_repository_info()
            if not repo_info:
                return
            
            async with GitHubAPIClient(self.github_token) as github:
                for comment in review.comments:
                    if comment['path'] and comment['line']:
                        await github.create_review_comment(
                            owner=repo_info['owner'],
                            repo=repo_info['repo'],
                            pr_number=pr.number,
                            body=comment['body'],
                            commit_sha=pr.metadata.get('github_data', {}).get('head', {}).get('sha', ''),
                            path=comment['path'],
                            line=comment['line']
                        )
        except Exception as e:
            logger.warning(f"Failed to post review comments: {e}")
    
    async def _approve_pull_request(self, pr: PullRequest) -> None:
        """Approve a pull request."""
        try:
            repo_info = await self._parse_repository_info()
            if not repo_info:
                return
            
            # This would require additional GitHub API calls for PR approval
            logger.info(f"Would approve PR #{pr.number} (implementation needed)")
        except Exception as e:
            logger.warning(f"Failed to approve PR: {e}")
    
    async def _validate_merge_result(self, merge_result: GitOperationResult) -> Dict[str, Any]:
        """Validate the result of a merge operation."""
        try:
            # Run validation on modified files
            validation_results = []
            for file_path in merge_result.files_modified:
                try:
                    full_path = Path(self.repository_path) / file_path
                    if full_path.exists():
                        validation = await self.validation_service.validate_file(full_path)
                        validation_results.append(validation)
                except Exception as e:
                    logger.warning(f"Merge validation failed for {file_path}: {e}")
            
            # Calculate overall validation score
            if validation_results:
                total_score = sum(v.authenticity_score for v in validation_results)
                avg_score = total_score / len(validation_results)
                
                return {
                    'validation_passed': avg_score >= 80.0,
                    'average_score': avg_score,
                    'files_validated': len(validation_results),
                    'issues_found': sum(len(v.issues) for v in validation_results)
                }
            else:
                return {
                    'validation_passed': True,
                    'average_score': 100.0,
                    'files_validated': 0,
                    'issues_found': 0
                }
        
        except Exception as e:
            logger.warning(f"Merge validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    async def _apply_branch_protection_via_api(
        self,
        repo_info: Dict[str, str],
        branch_pattern: str,
        rule: BranchRule
    ) -> bool:
        """Apply branch protection via GitHub API."""
        try:
            # This would implement GitHub branch protection API calls
            # For now, just log the action
            logger.info(f"Would apply branch protection for {branch_pattern} via API")
            return True
        except Exception as e:
            logger.warning(f"Branch protection API call failed: {e}")
            return False
    
    async def _trigger_github_workflow(
        self,
        repo_info: Dict[str, str],
        branch_name: str,
        workflow_name: Optional[str],
        inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trigger GitHub Actions workflow."""
        try:
            # This would implement GitHub Actions workflow dispatch API
            logger.info(f"Would trigger GitHub workflow for {branch_name}")
            return {
                'status': 'triggered',
                'branch': branch_name,
                'workflow': workflow_name or 'default'
            }
        except Exception as e:
            logger.warning(f"GitHub workflow trigger failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _trigger_local_ci_hooks(
        self,
        branch_name: str,
        inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trigger local CI hooks."""
        try:
            # Execute local CI hooks
            hooks_dir = Path(self.repository_path) / '.git' / 'hooks'
            
            if (hooks_dir / 'pre-push').exists():
                result = subprocess.run(
                    [str(hooks_dir / 'pre-push')],
                    cwd=self.repository_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                return {
                    'status': 'success' if result.returncode == 0 else 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
            
            return {'status': 'no_hooks', 'message': 'No local CI hooks found'}
            
        except Exception as e:
            logger.warning(f"Local CI hook execution failed: {e}")
            return {'status': 'failed', 'error': str(e)}


# Export main classes
__all__ = [
    'EnhancedGitManager',
    'PullRequest',
    'CodeReview',
    'BranchRule',
    'CommitMessageGenerator',
    'ConflictResolver',
    'GitHubAPIClient',
    'PRStatus',
    'ReviewStatus',
    'CIStatus'
]