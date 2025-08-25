"""
Git Manager Integration Module

Provides comprehensive Git operations management including:
- Advanced Git operations with safety checks
- Branch management and strategy enforcement
- Commit message validation and formatting
- Merge conflict resolution assistance
- Repository analysis and metrics
- Hooks integration for workflow coordination
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
from typing import Any, Dict, List, Optional, Set, Union, Tuple
import tempfile
import hashlib

# Optional Git imports with fallbacks
try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    from git.objects import Commit, Blob, Tree
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False
    # Fallback classes
    class Repo:
        def __init__(self, *args, **kwargs):
            pass
    class InvalidGitRepositoryError(Exception):
        pass
    class GitCommandError(Exception):
        pass
    class Commit:
        pass
    class Blob:
        pass
    class Tree:
        pass

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Base exception for Git operation errors"""
    pass


class GitRepositoryError(GitError):
    """Raised when repository operations fail"""
    pass


class GitCommitError(GitError):
    """Raised when commit operations fail"""
    pass


class GitBranchError(GitError):
    """Raised when branch operations fail"""
    pass


class GitMergeError(GitError):
    """Raised when merge operations fail"""
    pass


class GitOperationStatus(Enum):
    """Git operation status"""
    SUCCESS = "success"
    FAILED = "failed"
    CONFLICT = "conflict"
    ABORTED = "aborted"
    PENDING = "pending"


class BranchStrategy(Enum):
    """Branch management strategies"""
    GITFLOW = "gitflow"
    GITHUB_FLOW = "github_flow"
    GITLAB_FLOW = "gitlab_flow"
    CUSTOM = "custom"


@dataclass
class GitOperationResult:
    """Result from Git operations"""
    operation: str
    status: GitOperationStatus
    message: str = ""
    files_modified: List[str] = field(default_factory=list)
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        return self.status == GitOperationStatus.SUCCESS
    
    @property
    def has_conflicts(self) -> bool:
        return self.status == GitOperationStatus.CONFLICT


@dataclass
class CommitInfo:
    """Commit information structure"""
    hash: str
    author: str
    email: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    parents: List[str]


@dataclass
class BranchInfo:
    """Branch information structure"""
    name: str
    is_current: bool
    is_remote: bool
    upstream: Optional[str]
    last_commit: Optional[CommitInfo]
    ahead: int = 0
    behind: int = 0


@dataclass
class RepositoryStats:
    """Repository statistics and metrics"""
    total_commits: int
    total_branches: int
    total_contributors: int
    lines_of_code: int
    file_count: int
    languages: Dict[str, int]
    commit_activity: Dict[str, int]
    branch_activity: Dict[str, int]
    last_activity: datetime


class GitManager:
    """
    Advanced Git operations manager with safety features
    
    Features:
    - Safe Git operations with rollback capabilities
    - Advanced branch management and strategies
    - Intelligent commit message validation
    - Merge conflict detection and resolution
    - Repository analysis and health checks
    - Integration with Claude Flow hooks
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        repository_path: Optional[Union[str, Path]] = None,
        branch_strategy: BranchStrategy = BranchStrategy.GITHUB_FLOW,
        auto_stage: bool = False,
        safe_mode: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        claude_flow_binary: str = "npx claude-flow@alpha"
    ):
        self.repository_path = Path(repository_path) if repository_path else Path.cwd()
        self.branch_strategy = branch_strategy
        self.auto_stage = auto_stage
        self.safe_mode = safe_mode
        self.max_file_size = max_file_size
        self.claude_flow_binary = claude_flow_binary
        
        # Initialize repository connection
        self.repo: Optional[Repo] = None
        self._initialize_repository()
        
        # Operation tracking
        self.operation_history: List[GitOperationResult] = []
        self.pending_operations: List[str] = []
        
        # Safety features
        self.backup_refs: Dict[str, str] = {}
        self.rollback_points: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_execution_time': 0.0,
            'files_processed': 0
        }
        
        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for Git operations"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _initialize_repository(self):
        """Initialize Git repository connection"""
        try:
            if self.repository_path.exists():
                self.repo = Repo(self.repository_path)
                logger.info(f"Initialized Git repository at {self.repository_path}")
            else:
                logger.warning(f"Repository path does not exist: {self.repository_path}")
        except InvalidGitRepositoryError:
            logger.warning(f"Not a valid Git repository: {self.repository_path}")
        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")

    async def initialize_repository(
        self,
        path: Optional[Union[str, Path]] = None,
        initial_branch: str = "main"
    ) -> GitOperationResult:
        """
        Initialize a new Git repository
        
        Args:
            path: Repository path (uses current if None)
            initial_branch: Name of initial branch
            
        Returns:
            GitOperationResult with operation status
        """
        start_time = time.time()
        
        if path:
            self.repository_path = Path(path)
        
        logger.info(f"Initializing Git repository at {self.repository_path}")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", "Initializing Git repository")
            
            # Create directory if it doesn't exist
            self.repository_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize repository
            self.repo = Repo.init(self.repository_path, initial_branch=initial_branch)
            
            # Set up basic configuration
            await self._setup_repository_config()
            
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="init",
                status=GitOperationStatus.SUCCESS,
                message=f"Repository initialized with branch '{initial_branch}'",
                branch=initial_branch,
                execution_time=execution_time,
                metadata={'initial_branch': initial_branch}
            )
            
            # Execute hooks post-task
            await self._execute_hook(
                "post-edit",
                "Git repository initialized",
                memory_key="git/repository/initialized"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Repository initialized successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="init",
                status=GitOperationStatus.FAILED,
                message=f"Repository initialization failed: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Repository initialization failed: {e}")
            raise GitRepositoryError(f"Repository initialization failed: {e}") from e

    async def add_files(
        self,
        files: Union[str, List[str], Path],
        validate_size: bool = True,
        ignore_patterns: Optional[List[str]] = None
    ) -> GitOperationResult:
        """
        Add files to Git staging area with safety checks
        
        Args:
            files: File(s) to add (string, list, or Path)
            validate_size: Whether to validate file sizes
            ignore_patterns: Patterns to ignore during add
            
        Returns:
            GitOperationResult with operation status
        """
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        start_time = time.time()
        
        # Normalize files input
        if isinstance(files, (str, Path)):
            file_list = [str(files)]
        else:
            file_list = [str(f) for f in files]
        
        logger.info(f"Adding {len(file_list)} files to Git staging")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Adding {len(file_list)} files to Git")
            
            # Validate files if enabled
            if validate_size or ignore_patterns:
                file_list = await self._validate_files_for_staging(
                    file_list, validate_size, ignore_patterns
                )
            
            # Create backup point if safe mode enabled
            if self.safe_mode:
                await self._create_rollback_point("before_add")
            
            # Add files to staging
            added_files = []
            for file_path in file_list:
                try:
                    self.repo.index.add([file_path])
                    added_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Failed to add {file_path}: {e}")
            
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="add",
                status=GitOperationStatus.SUCCESS,
                message=f"Added {len(added_files)} files to staging",
                files_modified=added_files,
                execution_time=execution_time,
                metadata={
                    'requested_files': len(file_list),
                    'added_files': len(added_files),
                    'skipped_files': len(file_list) - len(added_files)
                }
            )
            
            # Execute hooks post-edit
            for file_path in added_files:
                await self._execute_hook(
                    "post-edit",
                    f"File {file_path} staged",
                    memory_key=f"git/staging/{file_path}",
                    file_path=file_path
                )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Successfully added {len(added_files)} files in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="add",
                status=GitOperationStatus.FAILED,
                message=f"Failed to add files: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to add files: {e}")
            raise GitCommitError(f"Failed to add files: {e}") from e

    async def commit(
        self,
        message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
        validate_message: bool = True,
        auto_add: bool = False
    ) -> GitOperationResult:
        """
        Create a commit with enhanced message validation
        
        Args:
            message: Commit message
            author_name: Optional author name override
            author_email: Optional author email override
            validate_message: Whether to validate commit message format
            auto_add: Whether to automatically add modified files
            
        Returns:
            GitOperationResult with operation status
        """
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        start_time = time.time()
        
        logger.info(f"Creating commit with message: {message[:50]}...")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", "Creating Git commit")
            
            # Validate commit message
            if validate_message:
                validation_result = await self._validate_commit_message(message)
                if not validation_result['valid']:
                    raise GitCommitError(f"Invalid commit message: {validation_result['error']}")
            
            # Auto-add files if requested
            if auto_add:
                modified_files = [item.a_path for item in self.repo.index.diff(None)]
                untracked_files = self.repo.untracked_files
                
                if modified_files or untracked_files:
                    all_files = modified_files + untracked_files
                    await self.add_files(all_files)
            
            # Check if there are changes to commit
            if not self.repo.index.diff("HEAD"):
                if not self.repo.untracked_files:
                    result = GitOperationResult(
                        operation="commit",
                        status=GitOperationStatus.SUCCESS,
                        message="No changes to commit",
                        execution_time=time.time() - start_time
                    )
                    return result
            
            # Create backup point
            if self.safe_mode:
                await self._create_rollback_point("before_commit")
            
            # Set author information if provided
            actor = None
            if author_name and author_email:
                from git import Actor
                actor = Actor(author_name, author_email)
            
            # Create commit
            commit = self.repo.index.commit(
                message=message,
                author=actor,
                committer=actor
            )
            
            # Get commit information
            files_in_commit = []
            if commit.parents:
                # Get files changed in this commit
                diff = commit.diff(commit.parents[0])
                files_in_commit = [item.a_path or item.b_path for item in diff]
            else:
                # First commit - all tracked files
                files_in_commit = [item[0] for item in self.repo.index.entries.keys()]
            
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="commit",
                status=GitOperationStatus.SUCCESS,
                message=f"Committed changes: {message}",
                files_modified=files_in_commit,
                commit_hash=commit.hexsha,
                branch=self.repo.active_branch.name,
                execution_time=execution_time,
                metadata={
                    'commit_hash': commit.hexsha[:8],
                    'author': str(commit.author),
                    'files_changed': len(files_in_commit),
                    'message_length': len(message)
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"Commit created: {commit.hexsha[:8]}",
                memory_key=f"git/commits/{commit.hexsha[:8]}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Commit {commit.hexsha[:8]} created successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="commit",
                status=GitOperationStatus.FAILED,
                message=f"Failed to create commit: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to create commit: {e}")
            raise GitCommitError(f"Failed to create commit: {e}") from e

    async def create_branch(
        self,
        branch_name: str,
        from_branch: Optional[str] = None,
        checkout: bool = True,
        push_upstream: bool = False
    ) -> GitOperationResult:
        """
        Create a new branch with strategy validation
        
        Args:
            branch_name: Name of new branch
            from_branch: Source branch (current branch if None)
            checkout: Whether to checkout the new branch
            push_upstream: Whether to push branch to remote
            
        Returns:
            GitOperationResult with operation status
        """
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        start_time = time.time()
        
        logger.info(f"Creating branch '{branch_name}'")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Creating branch {branch_name}")
            
            # Validate branch name
            if not await self._validate_branch_name(branch_name):
                raise GitBranchError(f"Invalid branch name: {branch_name}")
            
            # Check if branch already exists
            existing_branches = [ref.name.split('/')[-1] for ref in self.repo.refs]
            if branch_name in existing_branches:
                raise GitBranchError(f"Branch '{branch_name}' already exists")
            
            # Determine source commit
            if from_branch:
                try:
                    source_commit = self.repo.commit(from_branch)
                except:
                    raise GitBranchError(f"Source branch '{from_branch}' not found")
            else:
                source_commit = self.repo.head.commit
            
            # Create backup point
            if self.safe_mode:
                await self._create_rollback_point("before_branch_create")
            
            # Create branch
            new_branch = self.repo.create_head(branch_name, source_commit)
            
            # Checkout if requested
            current_branch = self.repo.active_branch.name
            if checkout:
                new_branch.checkout()
            
            # Push upstream if requested
            if push_upstream and 'origin' in [remote.name for remote in self.repo.remotes]:
                try:
                    origin = self.repo.remote('origin')
                    origin.push(new_branch.name)
                except Exception as e:
                    logger.warning(f"Failed to push branch upstream: {e}")
            
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="create_branch",
                status=GitOperationStatus.SUCCESS,
                message=f"Branch '{branch_name}' created successfully",
                branch=new_branch.name,
                execution_time=execution_time,
                metadata={
                    'from_branch': from_branch or current_branch,
                    'checked_out': checkout,
                    'pushed_upstream': push_upstream,
                    'source_commit': source_commit.hexsha[:8]
                }
            )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"Branch {branch_name} created",
                memory_key=f"git/branches/{branch_name}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Branch '{branch_name}' created successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="create_branch",
                status=GitOperationStatus.FAILED,
                message=f"Failed to create branch: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Failed to create branch '{branch_name}': {e}")
            raise GitBranchError(f"Failed to create branch: {e}") from e

    async def merge_branch(
        self,
        branch_name: str,
        strategy: str = "recursive",
        no_ff: bool = False,
        squash: bool = False
    ) -> GitOperationResult:
        """
        Merge branch with conflict detection and resolution assistance
        
        Args:
            branch_name: Branch to merge
            strategy: Merge strategy
            no_ff: Force no fast-forward merge
            squash: Squash commits during merge
            
        Returns:
            GitOperationResult with operation status
        """
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        start_time = time.time()
        current_branch = self.repo.active_branch.name
        
        logger.info(f"Merging branch '{branch_name}' into '{current_branch}'")
        
        try:
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Merging branch {branch_name}")
            
            # Validate target branch exists
            try:
                target_branch = self.repo.heads[branch_name]
            except IndexError:
                raise GitBranchError(f"Branch '{branch_name}' not found")
            
            # Create backup point
            if self.safe_mode:
                await self._create_rollback_point("before_merge")
            
            # Check for potential conflicts before merge
            conflicts = await self._detect_potential_conflicts(branch_name)
            
            # Perform merge
            merge_base = self.repo.merge_base(self.repo.head.commit, target_branch.commit)
            
            if not merge_base:
                raise GitMergeError("No common ancestor found for merge")
            
            # Execute merge
            try:
                if squash:
                    # Squash merge
                    self.repo.git.merge(branch_name, squash=True)
                    merge_commit = None
                else:
                    # Regular merge
                    merge_commit = self.repo.index.merge_tree(
                        target_branch.commit, 
                        base=merge_base[0]
                    )
                    
                    if no_ff:
                        self.repo.git.merge(branch_name, no_ff=True)
                    else:
                        self.repo.git.merge(branch_name)
                
                # Check for conflicts after merge
                unmerged_files = [item[0] for item in self.repo.index.unmerged_blobs()]
                
                if unmerged_files:
                    result = GitOperationResult(
                        operation="merge",
                        status=GitOperationStatus.CONFLICT,
                        message=f"Merge conflicts detected in {len(unmerged_files)} files",
                        branch=current_branch,
                        conflicts=unmerged_files,
                        execution_time=time.time() - start_time,
                        metadata={
                            'target_branch': branch_name,
                            'conflict_files': unmerged_files,
                            'merge_strategy': strategy
                        }
                    )
                else:
                    # Successful merge
                    latest_commit = self.repo.head.commit
                    
                    result = GitOperationResult(
                        operation="merge",
                        status=GitOperationStatus.SUCCESS,
                        message=f"Successfully merged '{branch_name}' into '{current_branch}'",
                        commit_hash=latest_commit.hexsha,
                        branch=current_branch,
                        execution_time=time.time() - start_time,
                        metadata={
                            'target_branch': branch_name,
                            'merge_strategy': strategy,
                            'squashed': squash,
                            'fast_forward': not no_ff
                        }
                    )
                
            except GitCommandError as e:
                # Handle merge conflicts
                unmerged_files = [item[0] for item in self.repo.index.unmerged_blobs()]
                
                result = GitOperationResult(
                    operation="merge",
                    status=GitOperationStatus.CONFLICT,
                    message=f"Merge failed with conflicts: {e}",
                    branch=current_branch,
                    conflicts=unmerged_files,
                    execution_time=time.time() - start_time,
                    metadata={
                        'target_branch': branch_name,
                        'error': str(e),
                        'conflict_files': unmerged_files
                    }
                )
            
            # Execute hooks post-edit
            await self._execute_hook(
                "post-edit",
                f"Merge operation completed: {result.status.value}",
                memory_key=f"git/merges/{branch_name}_{int(time.time())}"
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            if result.status == GitOperationStatus.CONFLICT:
                logger.warning(f"Merge conflicts detected in {len(result.conflicts)} files")
            else:
                logger.info(f"Merge completed successfully in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="merge",
                status=GitOperationStatus.FAILED,
                message=f"Merge failed: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Merge failed: {e}")
            raise GitMergeError(f"Merge failed: {e}") from e

    async def get_repository_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status and metrics"""
        
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        try:
            # Basic repository info
            current_branch = self.repo.active_branch.name if not self.repo.head.is_detached else None
            
            # File status
            modified_files = [item.a_path for item in self.repo.index.diff(None)]
            staged_files = [item.a_path for item in self.repo.index.diff("HEAD")]
            untracked_files = list(self.repo.untracked_files)
            
            # Branch information
            branches = []
            for branch in self.repo.heads:
                branch_info = BranchInfo(
                    name=branch.name,
                    is_current=branch.name == current_branch,
                    is_remote=False,
                    upstream=None,
                    last_commit=self._commit_to_info(branch.commit) if branch.commit else None
                )
                branches.append(branch_info)
            
            # Remote branches
            for remote in self.repo.remotes:
                for ref in remote.refs:
                    if ref.name != f"{remote.name}/HEAD":
                        branch_name = ref.name.split('/')[-1]
                        branch_info = BranchInfo(
                            name=f"{remote.name}/{branch_name}",
                            is_current=False,
                            is_remote=True,
                            upstream=remote.name,
                            last_commit=self._commit_to_info(ref.commit) if ref.commit else None
                        )
                        branches.append(branch_info)
            
            # Recent commits
            recent_commits = []
            for commit in self.repo.iter_commits(max_count=10):
                recent_commits.append(self._commit_to_info(commit))
            
            # Repository statistics
            stats = await self._calculate_repository_stats()
            
            return {
                'repository_path': str(self.repository_path),
                'current_branch': current_branch,
                'is_dirty': self.repo.is_dirty(),
                'files': {
                    'modified': modified_files,
                    'staged': staged_files,
                    'untracked': untracked_files,
                    'total_tracked': len(list(self.repo.git.ls_files().splitlines()))
                },
                'branches': [
                    {
                        'name': b.name,
                        'is_current': b.is_current,
                        'is_remote': b.is_remote,
                        'upstream': b.upstream,
                        'last_commit_hash': b.last_commit.hash[:8] if b.last_commit else None,
                        'last_commit_message': b.last_commit.message if b.last_commit else None
                    }
                    for b in branches
                ],
                'recent_commits': [
                    {
                        'hash': c.hash[:8],
                        'author': c.author,
                        'message': c.message,
                        'timestamp': c.timestamp.isoformat(),
                        'files_changed': len(c.files_changed)
                    }
                    for c in recent_commits
                ],
                'statistics': {
                    'total_commits': stats.total_commits,
                    'total_branches': stats.total_branches,
                    'total_contributors': stats.total_contributors,
                    'lines_of_code': stats.lines_of_code,
                    'file_count': stats.file_count,
                    'languages': stats.languages,
                    'last_activity': stats.last_activity.isoformat()
                },
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Failed to get repository status: {e}")
            raise GitRepositoryError(f"Failed to get repository status: {e}") from e

    async def resolve_conflicts(
        self,
        files: Optional[List[str]] = None,
        resolution_strategy: str = "manual"
    ) -> GitOperationResult:
        """
        Assist with merge conflict resolution
        
        Args:
            files: Specific files to resolve (all if None)
            resolution_strategy: Resolution strategy (manual, ours, theirs, auto)
            
        Returns:
            GitOperationResult with resolution status
        """
        if not self.repo:
            raise GitRepositoryError("Repository not initialized")
        
        start_time = time.time()
        
        try:
            # Get unmerged files
            unmerged_blobs = self.repo.index.unmerged_blobs()
            conflict_files = list(unmerged_blobs.keys())
            
            if not conflict_files:
                return GitOperationResult(
                    operation="resolve_conflicts",
                    status=GitOperationStatus.SUCCESS,
                    message="No conflicts to resolve",
                    execution_time=time.time() - start_time
                )
            
            if files:
                # Filter to specified files
                conflict_files = [f for f in conflict_files if f in files]
            
            logger.info(f"Resolving conflicts in {len(conflict_files)} files")
            
            # Execute hooks pre-task
            await self._execute_hook("pre-task", f"Resolving conflicts in {len(conflict_files)} files")
            
            resolved_files = []
            failed_files = []
            
            for file_path in conflict_files:
                try:
                    if resolution_strategy == "ours":
                        # Accept our version
                        self.repo.git.checkout("--ours", file_path)
                        self.repo.index.add([file_path])
                        resolved_files.append(file_path)
                        
                    elif resolution_strategy == "theirs":
                        # Accept their version
                        self.repo.git.checkout("--theirs", file_path)
                        self.repo.index.add([file_path])
                        resolved_files.append(file_path)
                        
                    elif resolution_strategy == "auto":
                        # Attempt automatic resolution
                        success = await self._attempt_auto_resolution(file_path)
                        if success:
                            resolved_files.append(file_path)
                        else:
                            failed_files.append(file_path)
                            
                    else:  # manual
                        # Provide conflict information for manual resolution
                        conflict_info = await self._analyze_conflict(file_path)
                        failed_files.append({
                            'file': file_path,
                            'conflict_info': conflict_info
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to resolve conflict in {file_path}: {e}")
                    failed_files.append(file_path)
            
            execution_time = time.time() - start_time
            
            if resolved_files and not failed_files:
                status = GitOperationStatus.SUCCESS
                message = f"Resolved conflicts in {len(resolved_files)} files"
            elif resolved_files and failed_files:
                status = GitOperationStatus.SUCCESS  # Partial success
                message = f"Resolved {len(resolved_files)} files, {len(failed_files)} require manual intervention"
            else:
                status = GitOperationStatus.FAILED
                message = f"Failed to resolve conflicts in {len(failed_files)} files"
            
            result = GitOperationResult(
                operation="resolve_conflicts",
                status=status,
                message=message,
                files_modified=resolved_files,
                execution_time=execution_time,
                metadata={
                    'resolution_strategy': resolution_strategy,
                    'resolved_files': resolved_files,
                    'failed_files': failed_files,
                    'total_conflicts': len(conflict_files)
                }
            )
            
            # Execute hooks post-edit
            for file_path in resolved_files:
                await self._execute_hook(
                    "post-edit",
                    f"Conflict resolved in {file_path}",
                    memory_key=f"git/conflicts_resolved/{file_path}",
                    file_path=file_path
                )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.info(f"Conflict resolution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = GitOperationResult(
                operation="resolve_conflicts",
                status=GitOperationStatus.FAILED,
                message=f"Conflict resolution failed: {e}",
                execution_time=execution_time
            )
            
            self._update_metrics(result)
            self.operation_history.append(result)
            
            logger.error(f"Conflict resolution failed: {e}")
            raise GitMergeError(f"Conflict resolution failed: {e}") from e

    async def _validate_files_for_staging(
        self,
        files: List[str],
        validate_size: bool,
        ignore_patterns: Optional[List[str]]
    ) -> List[str]:
        """Validate files before adding to staging"""
        
        valid_files = []
        
        for file_path in files:
            full_path = self.repository_path / file_path
            
            # Check if file exists
            if not full_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Check ignore patterns
            if ignore_patterns:
                should_ignore = False
                for pattern in ignore_patterns:
                    if re.search(pattern, file_path):
                        logger.info(f"Ignoring file {file_path} (matches pattern: {pattern})")
                        should_ignore = True
                        break
                if should_ignore:
                    continue
            
            # Check file size
            if validate_size and full_path.is_file():
                file_size = full_path.stat().st_size
                if file_size > self.max_file_size:
                    logger.warning(f"File {file_path} too large ({file_size} bytes), skipping")
                    continue
            
            valid_files.append(file_path)
        
        return valid_files

    async def _validate_commit_message(self, message: str) -> Dict[str, Any]:
        """Validate commit message format and content"""
        
        # Basic validation rules
        if len(message) < 10:
            return {'valid': False, 'error': 'Commit message too short (minimum 10 characters)'}
        
        if len(message.splitlines()[0]) > 72:
            return {'valid': False, 'error': 'First line too long (maximum 72 characters)'}
        
        # Check for common patterns
        first_line = message.splitlines()[0]
        
        # Should start with a verb in imperative mood
        imperative_starters = [
            'add', 'fix', 'update', 'remove', 'refactor', 'improve', 'enhance',
            'create', 'implement', 'optimize', 'clean', 'format', 'document'
        ]
        
        starts_with_imperative = any(
            first_line.lower().startswith(verb) for verb in imperative_starters
        )
        
        if not starts_with_imperative:
            logger.warning(f"Commit message doesn't start with imperative verb: {first_line}")
        
        # Should not end with period
        if first_line.endswith('.'):
            return {'valid': False, 'error': 'First line should not end with period'}
        
        return {
            'valid': True,
            'suggestions': [] if starts_with_imperative else ['Consider starting with an imperative verb']
        }

    async def _validate_branch_name(self, branch_name: str) -> bool:
        """Validate branch name according to Git standards and strategy"""
        
        # Basic Git rules
        if not re.match(r'^[a-zA-Z0-9._/-]+$', branch_name):
            return False
        
        if branch_name.startswith('-') or branch_name.endswith('.lock'):
            return False
        
        if '..' in branch_name or branch_name.startswith('/') or branch_name.endswith('/'):
            return False
        
        # Strategy-specific validation
        if self.branch_strategy == BranchStrategy.GITFLOW:
            valid_prefixes = ['feature/', 'hotfix/', 'release/', 'develop', 'main', 'master']
            if not any(branch_name.startswith(prefix) for prefix in valid_prefixes):
                logger.warning(f"Branch name '{branch_name}' doesn't follow GitFlow convention")
        
        return True

    async def _detect_potential_conflicts(self, branch_name: str) -> List[str]:
        """Detect potential conflicts before merge"""
        
        try:
            target_branch = self.repo.heads[branch_name]
            current_commit = self.repo.head.commit
            target_commit = target_branch.commit
            
            # Find merge base
            merge_base = self.repo.merge_base(current_commit, target_commit)
            if not merge_base:
                return []
            
            # Get files changed in both branches
            current_changes = {item.a_path for item in current_commit.diff(merge_base[0])}
            target_changes = {item.a_path for item in target_commit.diff(merge_base[0])}
            
            # Files changed in both branches are potential conflicts
            potential_conflicts = list(current_changes.intersection(target_changes))
            
            return potential_conflicts
            
        except Exception as e:
            logger.warning(f"Could not detect potential conflicts: {e}")
            return []

    def _commit_to_info(self, commit: Commit) -> CommitInfo:
        """Convert Git commit to CommitInfo structure"""
        
        # Get files changed in commit
        files_changed = []
        insertions = 0
        deletions = 0
        
        if commit.parents:
            diff = commit.diff(commit.parents[0])
            files_changed = [item.a_path or item.b_path for item in diff]
            
            # Calculate insertions and deletions
            for item in diff:
                if hasattr(item, 'change_type'):
                    if item.change_type in ['A', 'M']:  # Added or Modified
                        try:
                            lines = item.b_blob.data_stream.read().decode('utf-8', errors='ignore').splitlines()
                            insertions += len(lines)
                        except:
                            pass
                    if item.change_type in ['D', 'M']:  # Deleted or Modified
                        try:
                            lines = item.a_blob.data_stream.read().decode('utf-8', errors='ignore').splitlines()
                            deletions += len(lines)
                        except:
                            pass
        
        return CommitInfo(
            hash=commit.hexsha,
            author=commit.author.name,
            email=commit.author.email,
            timestamp=datetime.fromtimestamp(commit.committed_date),
            message=commit.message.strip(),
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            parents=[parent.hexsha for parent in commit.parents]
        )

    async def _calculate_repository_stats(self) -> RepositoryStats:
        """Calculate comprehensive repository statistics"""
        
        try:
            # Basic counts
            total_commits = sum(1 for _ in self.repo.iter_commits())
            total_branches = len(list(self.repo.heads))
            
            # Contributors
            contributors = set()
            for commit in self.repo.iter_commits():
                contributors.add(commit.author.email)
            
            # File analysis
            file_count = 0
            lines_of_code = 0
            languages = {}
            
            for item in self.repo.head.commit.tree.traverse():
                if item.type == 'blob':  # File
                    file_count += 1
                    
                    # Detect language by extension
                    file_path = Path(item.path)
                    extension = file_path.suffix.lower()
                    
                    language_map = {
                        '.py': 'Python',
                        '.js': 'JavaScript',
                        '.ts': 'TypeScript',
                        '.java': 'Java',
                        '.cpp': 'C++',
                        '.c': 'C',
                        '.go': 'Go',
                        '.rs': 'Rust',
                        '.php': 'PHP',
                        '.rb': 'Ruby'
                    }
                    
                    language = language_map.get(extension, 'Other')
                    languages[language] = languages.get(language, 0) + 1
                    
                    # Count lines of code
                    try:
                        content = item.data_stream.read().decode('utf-8', errors='ignore')
                        lines_of_code += len(content.splitlines())
                    except:
                        pass
            
            # Activity analysis
            commit_activity = {}
            branch_activity = {}
            
            for commit in self.repo.iter_commits(max_count=100):
                # Daily activity
                day_key = commit.committed_datetime.strftime('%Y-%m-%d')
                commit_activity[day_key] = commit_activity.get(day_key, 0) + 1
            
            # Last activity
            last_commit = next(self.repo.iter_commits(max_count=1))
            last_activity = datetime.fromtimestamp(last_commit.committed_date)
            
            return RepositoryStats(
                total_commits=total_commits,
                total_branches=total_branches,
                total_contributors=len(contributors),
                lines_of_code=lines_of_code,
                file_count=file_count,
                languages=languages,
                commit_activity=commit_activity,
                branch_activity=branch_activity,
                last_activity=last_activity
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate repository stats: {e}")
            return RepositoryStats(
                total_commits=0,
                total_branches=0,
                total_contributors=0,
                lines_of_code=0,
                file_count=0,
                languages={},
                commit_activity={},
                branch_activity={},
                last_activity=datetime.utcnow()
            )

    async def _attempt_auto_resolution(self, file_path: str) -> bool:
        """Attempt automatic conflict resolution using simple heuristics"""
        
        try:
            # Read conflict file
            full_path = self.repository_path / file_path
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            # Parse conflict markers
            conflict_pattern = r'<<<<<<< .*?\n(.*?)\n=======\n(.*?)\n>>>>>>> .*?\n'
            conflicts = re.findall(conflict_pattern, content, re.DOTALL)
            
            if not conflicts:
                return False
            
            resolved_content = content
            
            for ours, theirs in conflicts:
                # Simple resolution heuristics
                if len(ours.strip()) == 0:
                    # Keep theirs if ours is empty
                    resolved_content = resolved_content.replace(
                        f'<<<<<<< HEAD\n{ours}\n=======\n{theirs}\n>>>>>>> ',
                        theirs
                    )
                elif len(theirs.strip()) == 0:
                    # Keep ours if theirs is empty
                    resolved_content = resolved_content.replace(
                        f'<<<<<<< HEAD\n{ours}\n=======\n{theirs}\n>>>>>>> ',
                        ours
                    )
                elif ours == theirs:
                    # Identical content
                    resolved_content = resolved_content.replace(
                        f'<<<<<<< HEAD\n{ours}\n=======\n{theirs}\n>>>>>>> ',
                        ours
                    )
                else:
                    # Cannot auto-resolve
                    return False
            
            # Write resolved content
            full_path.write_text(resolved_content, encoding='utf-8')
            
            # Add to index
            self.repo.index.add([file_path])
            
            return True
            
        except Exception as e:
            logger.error(f"Auto-resolution failed for {file_path}: {e}")
            return False

    async def _analyze_conflict(self, file_path: str) -> Dict[str, Any]:
        """Analyze conflict for manual resolution assistance"""
        
        try:
            full_path = self.repository_path / file_path
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            # Find conflict markers
            lines = content.splitlines()
            conflicts = []
            
            i = 0
            while i < len(lines):
                if lines[i].startswith('<<<<<<<'):
                    # Found conflict start
                    conflict_start = i
                    ours_lines = []
                    theirs_lines = []
                    
                    # Find separator
                    separator_line = None
                    for j in range(i + 1, len(lines)):
                        if lines[j] == '=======':
                            separator_line = j
                            break
                        ours_lines.append(lines[j])
                    
                    if separator_line is None:
                        i += 1
                        continue
                    
                    # Find conflict end
                    conflict_end = None
                    for j in range(separator_line + 1, len(lines)):
                        if lines[j].startswith('>>>>>>>'):
                            conflict_end = j
                            break
                        theirs_lines.append(lines[j])
                    
                    if conflict_end is None:
                        i += 1
                        continue
                    
                    conflicts.append({
                        'start_line': conflict_start + 1,
                        'end_line': conflict_end + 1,
                        'ours': '\n'.join(ours_lines),
                        'theirs': '\n'.join(theirs_lines),
                        'ours_lines': len(ours_lines),
                        'theirs_lines': len(theirs_lines)
                    })
                    
                    i = conflict_end + 1
                else:
                    i += 1
            
            return {
                'file_path': file_path,
                'total_conflicts': len(conflicts),
                'conflicts': conflicts,
                'file_size': len(content),
                'total_lines': len(lines)
            }
            
        except Exception as e:
            logger.error(f"Conflict analysis failed for {file_path}: {e}")
            return {'error': str(e)}

    async def _setup_repository_config(self):
        """Setup basic repository configuration"""
        
        try:
            config = self.repo.config_writer()
            
            # Set default branch
            config.set_value('init', 'defaultBranch', 'main')
            
            # Set merge strategy
            config.set_value('merge', 'tool', 'vimdiff')
            
            # Set push default
            config.set_value('push', 'default', 'simple')
            
            # Release writer
            config.release()
            
            logger.info("Repository configuration completed")
            
        except Exception as e:
            logger.warning(f"Could not setup repository configuration: {e}")

    async def _create_rollback_point(self, operation: str):
        """Create rollback point for safe operations"""
        
        try:
            current_commit = self.repo.head.commit.hexsha
            current_branch = self.repo.active_branch.name
            
            rollback_point = {
                'operation': operation,
                'timestamp': datetime.utcnow(),
                'commit': current_commit,
                'branch': current_branch,
                'working_tree_clean': not self.repo.is_dirty()
            }
            
            self.rollback_points.append(rollback_point)
            
            # Limit rollback points to last 10
            if len(self.rollback_points) > 10:
                self.rollback_points = self.rollback_points[-10:]
            
            logger.debug(f"Created rollback point for {operation}")
            
        except Exception as e:
            logger.warning(f"Could not create rollback point: {e}")

    async def _execute_hook(
        self,
        hook_type: str,
        message: str,
        memory_key: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        """Execute Claude Flow hooks for coordination"""
        
        try:
            if hook_type == "pre-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "pre-task",
                    "--description", message
                ]
            elif hook_type == "post-task":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-task",
                    "--task-id", "git-operation"
                ]
            elif hook_type == "post-edit" and memory_key:
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "post-edit",
                    "--memory-key", memory_key
                ]
                if file_path:
                    cmd.extend(["--file", file_path])
            elif hook_type == "notify":
                cmd = [
                    *self.claude_flow_binary.split(),
                    "hooks", "notify",
                    "--message", message
                ]
            else:
                return
            
            # Execute hook command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(process.communicate(), timeout=10)
            
        except Exception as e:
            logger.debug(f"Hook execution failed ({hook_type}): {e}")

    def _update_metrics(self, result: GitOperationResult):
        """Update performance metrics"""
        
        self.metrics['total_operations'] += 1
        self.metrics['files_processed'] += len(result.files_modified)
        
        if result.is_success:
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['failed_operations'] += 1
        
        # Update average execution time
        total_time = (
            self.metrics['average_execution_time'] * (self.metrics['total_operations'] - 1) +
            result.execution_time
        )
        self.metrics['average_execution_time'] = total_time / self.metrics['total_operations']

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        success_rate = 0.0
        if self.metrics['total_operations'] > 0:
            success_rate = self.metrics['successful_operations'] / self.metrics['total_operations']
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'operations_history_size': len(self.operation_history),
            'rollback_points_available': len(self.rollback_points)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Git integration"""
        
        try:
            # Check repository status
            if not self.repo:
                return {
                    'status': 'unhealthy',
                    'error': 'Repository not initialized'
                }
            
            # Check Git executable
            result = await asyncio.create_subprocess_exec(
                'git', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            git_version = stdout.decode().strip() if result.returncode == 0 else 'unknown'
            
            # Check repository integrity
            try:
                # Quick integrity check
                self.repo.git.fsck('--quick')
                integrity_ok = True
            except:
                integrity_ok = False
            
            return {
                'status': 'healthy',
                'repository_path': str(self.repository_path),
                'git_version': git_version,
                'current_branch': self.repo.active_branch.name if not self.repo.head.is_detached else 'detached',
                'is_dirty': self.repo.is_dirty(),
                'integrity_ok': integrity_ok,
                'metrics': self.get_metrics(),
                'active_operations': len(self.pending_operations),
                'rollback_points': len(self.rollback_points)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'repository_path': str(self.repository_path)
            }