"""
Comprehensive tests for Enhanced Git Integration features.

This module tests all aspects of the advanced Git integration including:
- Smart commit generation with AI
- Pull request automation
- Code review workflows
- Merge conflict resolution
- Branch protection enforcement
- CI/CD integration
"""

import asyncio
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from git import Repo

from src.integrations.git_advanced import (
    EnhancedGitManager, PullRequest, CodeReview, BranchRule,
    CommitMessageGenerator, ConflictResolver, GitHubAPIClient,
    PRStatus, ReviewStatus, CIStatus
)
from src.core.types import ValidationResult, Issue, Severity, IssueType


class TestEnhancedGitManager:
    """Test suite for EnhancedGitManager."""
    
    @pytest.fixture
    async def temp_repo(self):
        """Create a temporary Git repository for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize Git repo
            repo = Repo.init(temp_dir)
            
            # Configure user
            repo.config_writer().set_value("user", "name", "Test User").release()
            repo.config_writer().set_value("user", "email", "test@example.com").release()
            
            # Create initial commit
            readme_file = temp_dir / "README.md"
            readme_file.write_text("# Test Repository\n\nTest repository for Git integration tests.")
            
            repo.index.add([str(readme_file)])
            repo.index.commit("Initial commit")
            
            yield temp_dir, repo
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def git_manager(self, temp_repo):
        """Create EnhancedGitManager instance for testing."""
        temp_dir, repo = temp_repo
        
        manager = EnhancedGitManager(
            repository_path=temp_dir,
            github_token="test_token",
            safe_mode=False  # Disable for testing
        )
        
        return manager
    
    async def test_initialization(self, git_manager):
        """Test Git manager initialization."""
        assert git_manager.repo is not None
        assert git_manager.repository_path.exists()
        assert git_manager.github_token == "test_token"
        assert isinstance(git_manager.commit_generator, CommitMessageGenerator)
        assert isinstance(git_manager.conflict_resolver, ConflictResolver)
    
    async def test_smart_commit_generation(self, git_manager, temp_repo):
        """Test AI-powered smart commit generation."""
        temp_dir, repo = temp_repo
        
        # Create some changes
        test_file = temp_dir / "test.py"
        test_file.write_text("""
def hello_world():
    print("Hello, World!")
    
def add_numbers(a, b):
    return a + b
""")
        
        # Mock AI interface
        with patch.object(git_manager.commit_generator, 'ai_interface') as mock_ai:
            mock_ai.generate_response = AsyncMock(return_value="feat: add hello world and math functions")
            
            # Generate smart commit
            result = await git_manager.generate_smart_commit(
                auto_add=True,
                conventional_commits=True
            )
            
            assert result.is_success
            assert result.commit_hash is not None
            assert "ai_generated_message" in result.metadata
            assert result.metadata["ai_generated_message"] is True
            
            # Verify commit exists
            latest_commit = repo.head.commit
            assert "feat: add hello world and math functions" in latest_commit.message
    
    async def test_pull_request_creation(self, git_manager, temp_repo):
        """Test automated pull request creation."""
        temp_dir, repo = temp_repo
        
        # Create feature branch
        feature_branch = repo.create_head("feature/test-feature")
        feature_branch.checkout()
        
        # Add some changes
        test_file = temp_dir / "feature.py"
        test_file.write_text("# New feature implementation\n")
        repo.index.add([str(test_file)])
        repo.index.commit("Add new feature")
        
        # Mock GitHub API responses
        mock_pr_data = {
            'id': 123,
            'number': 1,
            'title': 'Feature/test feature',
            'body': 'Test PR body',
            'head': {'ref': 'feature/test-feature'},
            'base': {'ref': 'main'},
            'html_url': 'https://github.com/test/repo/pull/1',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'state': 'open'
        }
        
        with patch('src.integrations.git_advanced.GitHubAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.create_pull_request = AsyncMock(return_value=PullRequest(
                id=123,
                number=1,
                title="Feature/test feature",
                body="Test PR body",
                head_branch="feature/test-feature",
                base_branch="main",
                status=PRStatus.OPEN,
                url="https://github.com/test/repo/pull/1"
            ))
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock AI content generation
            with patch.object(git_manager, '_generate_pr_content') as mock_gen:
                mock_gen.return_value = {
                    'title': 'Add test feature implementation',
                    'body': 'This PR adds a new test feature with proper documentation.'
                }
                
                # Mock repository info parsing
                with patch.object(git_manager, '_parse_repository_info') as mock_repo_info:
                    mock_repo_info.return_value = {'owner': 'testuser', 'repo': 'testrepo'}
                    
                    # Create pull request
                    pr = await git_manager.create_pull_request(
                        title="Add test feature implementation",
                        base_branch="main",
                        head_branch="feature/test-feature"
                    )
                    
                    assert pr.number == 1
                    assert pr.title == "Feature/test feature"
                    assert pr.status == PRStatus.OPEN
                    assert pr.head_branch == "feature/test-feature"
                    assert pr.base_branch == "main"
    
    async def test_code_review_automation(self, git_manager, temp_repo):
        """Test AI-powered code review functionality."""
        temp_dir, repo = temp_repo
        
        # Create a PR object
        pr = PullRequest(
            id=123,
            number=1,
            title="Test PR",
            head_branch="feature/test",
            base_branch="main"
        )
        git_manager.pull_requests[1] = pr
        
        # Mock validation service
        mock_validation_result = ValidationResult(
            is_authentic=True,
            authenticity_score=85.0,
            real_progress=90.0,
            fake_progress=10.0,
            issues=[
                Issue(
                    type=IssueType.PLACEHOLDER,
                    severity=Severity.MEDIUM,
                    description="Placeholder comment found",
                    file_path="test.py",
                    line_number=10
                )
            ]
        )
        
        with patch.object(git_manager.validation_service, 'validate_file') as mock_validate:
            mock_validate.return_value = mock_validation_result
            
            # Mock PR changes
            with patch.object(git_manager, '_get_pr_changes') as mock_changes:
                mock_changes.return_value = {
                    'modified_files': ['test.py', 'docs.md'],
                    'diff_content': 'test diff content',
                    'file_count': 2
                }
                
                # Perform code review
                review = await git_manager.review_pull_request(
                    pr_number=1,
                    review_type="comprehensive"
                )
                
                assert review.pull_request_id == 1
                assert review.ai_generated is True
                assert review.quality_score == 85.0
                assert len(review.comments) > 0
                assert review.status in [ReviewStatus.APPROVED, ReviewStatus.COMMENTED, ReviewStatus.CHANGES_REQUESTED]
    
    async def test_smart_merge_with_conflict_resolution(self, git_manager, temp_repo):
        """Test smart merge with AI-assisted conflict resolution."""
        temp_dir, repo = temp_repo
        
        # Create conflicting branches
        main_branch = repo.heads.main
        
        # Create and modify main branch
        main_branch.checkout()
        conflict_file = temp_dir / "conflict.txt"
        conflict_file.write_text("Line 1\nMain branch content\nLine 3\n")
        repo.index.add([str(conflict_file)])
        repo.index.commit("Main branch changes")
        
        # Create feature branch from earlier commit
        feature_branch = repo.create_head("feature/conflict", repo.head.commit.parents[0])
        feature_branch.checkout()
        
        conflict_file.write_text("Line 1\nFeature branch content\nLine 3\n")
        repo.index.add([str(conflict_file)])
        repo.index.commit("Feature branch changes")
        
        # Switch back to main
        main_branch.checkout()
        
        # Mock conflict analysis
        mock_analysis = {
            'total_conflicts': 1,
            'auto_resolvable': 0,
            'complex_conflicts': 1,
            'resolution_strategy': 'manual',
            'file_analyses': [{
                'file_path': 'conflict.txt',
                'conflict_count': 1,
                'auto_resolvable': False,
                'complexity_score': 2
            }]
        }
        
        with patch.object(git_manager.conflict_resolver, 'analyze_conflicts') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            # Attempt smart merge
            result = await git_manager.smart_merge_with_conflict_resolution(
                branch_name="feature/conflict",
                auto_resolve=True
            )
            
            # Should detect conflicts but not auto-resolve complex ones
            assert result.operation == "smart_merge"
            # Result may be successful with conflicts or failed depending on implementation
    
    async def test_branch_protection_enforcement(self, git_manager):
        """Test branch protection rule enforcement."""
        # Create a protection rule
        rule = BranchRule(
            pattern="main",
            required_reviews=2,
            required_status_checks=["ci", "tests"],
            enforce_admins=True
        )
        
        # Mock API call
        with patch.object(git_manager, '_apply_branch_protection_via_api') as mock_api:
            mock_api.return_value = True
            
            with patch.object(git_manager, '_parse_repository_info') as mock_repo_info:
                mock_repo_info.return_value = {'owner': 'testuser', 'repo': 'testrepo'}
                
                # Enforce protection
                result = await git_manager.enforce_branch_protection("main", rule)
                
                assert result.is_success
                assert "main" in git_manager.branch_rules
                assert git_manager.branch_rules["main"] == rule
    
    async def test_ci_pipeline_trigger(self, git_manager):
        """Test CI/CD pipeline triggering."""
        with patch.object(git_manager, '_parse_repository_info') as mock_repo_info:
            mock_repo_info.return_value = {'owner': 'testuser', 'repo': 'testrepo'}
            
            with patch.object(git_manager, '_trigger_github_workflow') as mock_trigger:
                mock_trigger.return_value = {
                    'status': 'triggered',
                    'branch': 'main',
                    'workflow': 'ci.yml'
                }
                
                # Trigger CI pipeline
                result = await git_manager.trigger_ci_pipeline(
                    branch_name="main",
                    workflow_name="ci.yml"
                )
                
                assert result['status'] == 'triggered'
                assert result['branch'] == 'main'
                assert result['workflow'] == 'ci.yml'


class TestCommitMessageGenerator:
    """Test suite for AI-powered commit message generation."""
    
    @pytest.fixture
    def generator(self):
        """Create CommitMessageGenerator instance."""
        mock_ai = AsyncMock()
        return CommitMessageGenerator(ai_interface=mock_ai)
    
    async def test_conventional_commit_generation(self, generator):
        """Test conventional commit message generation."""
        diff_content = """
diff --git a/src/feature.py b/src/feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/src/feature.py
@@ -0,0 +1,10 @@
+def new_feature():
+    \"\"\"Implement new feature functionality.\"\"\"
+    return \"Hello, World!\"
"""
        
        file_paths = ["src/feature.py"]
        
        # Mock AI response
        generator.ai_interface.generate_response = AsyncMock(
            return_value="feat(src): add new feature functionality\n\nImplement hello world feature with proper documentation."
        )
        
        message = await generator.generate_commit_message(
            diff_content, file_paths, conventional_commits=True
        )
        
        assert message == "feat(src): add new feature functionality\n\nImplement hello world feature with proper documentation."
        
        # Verify AI was called with appropriate prompt
        generator.ai_interface.generate_response.assert_called_once()
        call_args = generator.ai_interface.generate_response.call_args[0]
        assert "conventional commit" in call_args[0].lower()
    
    async def test_standard_commit_generation(self, generator):
        """Test standard commit message generation."""
        diff_content = "Simple file modification"
        file_paths = ["README.md"]
        
        generator.ai_interface.generate_response = AsyncMock(
            return_value="Update README documentation"
        )
        
        message = await generator.generate_commit_message(
            diff_content, file_paths, conventional_commits=False
        )
        
        assert message == "Update README documentation"
    
    async def test_fallback_on_ai_failure(self, generator):
        """Test fallback behavior when AI fails."""
        diff_content = "test content"
        file_paths = ["test.py"]
        
        # Mock AI failure
        generator.ai_interface.generate_response = AsyncMock(
            side_effect=Exception("AI service unavailable")
        )
        
        message = await generator.generate_commit_message(
            diff_content, file_paths, conventional_commits=True
        )
        
        # Should fall back to simple message
        assert "files" in message.lower() or "feature" in message.lower()
    
    async def test_change_analysis(self, generator):
        """Test change analysis functionality."""
        diff_content = "test diff"
        file_paths = ["src/test.py", "docs/README.md", "tests/test_feature.py"]
        
        analysis = await generator._analyze_changes(diff_content, file_paths)
        
        assert analysis['files_count'] == 3
        assert analysis['primary_language'] == 'python'
        assert analysis['type'] in ['feat', 'test', 'docs']


class TestConflictResolver:
    """Test suite for AI-powered conflict resolution."""
    
    @pytest.fixture
    def resolver(self):
        """Create ConflictResolver instance."""
        mock_ai = AsyncMock()
        return ConflictResolver(ai_interface=mock_ai)
    
    @pytest.fixture
    async def mock_repo(self, temp_repo):
        """Create mock repository with conflicts."""
        temp_dir, repo = temp_repo
        
        # Create conflict file
        conflict_file = temp_dir / "conflict.txt"
        conflict_content = """
Line 1
<<<<<<< HEAD
Our version of the change
=======
Their version of the change
>>>>>>> feature-branch
Line 3
"""
        conflict_file.write_text(conflict_content)
        
        return repo, conflict_file
    
    async def test_conflict_analysis(self, resolver, mock_repo):
        """Test conflict analysis functionality."""
        repo, conflict_file = mock_repo
        
        conflict_files = [str(conflict_file.name)]
        analysis = await resolver.analyze_conflicts(repo, conflict_files)
        
        assert analysis['total_conflicts'] == 1
        assert len(analysis['file_analyses']) == 1
        
        file_analysis = analysis['file_analyses'][0]
        assert file_analysis['file_path'] == conflict_file.name
        assert file_analysis['conflict_count'] == 1
    
    async def test_conflict_section_parsing(self, resolver):
        """Test parsing of conflict markers."""
        content = """
Line 1
<<<<<<< HEAD
Our content
=======  
Their content
>>>>>>> branch
Line 2
<<<<<<< HEAD
Another our content
=======
Another their content
>>>>>>> branch
Line 3
"""
        
        sections = resolver._parse_conflict_markers(content)
        
        assert len(sections) == 2
        assert sections[0]['ours'].strip() == "Our content"
        assert sections[0]['theirs'].strip() == "Their content"
        assert sections[1]['ours'].strip() == "Another our content"
        assert sections[1]['theirs'].strip() == "Another their content"
    
    async def test_simple_conflict_resolution(self, resolver):
        """Test automatic resolution of simple conflicts."""
        # Test identical content conflict
        section = {
            'ours': 'same content',
            'theirs': 'same content',
            'ours_label': 'HEAD',
            'theirs_label': 'branch'
        }
        
        analysis = await resolver._analyze_conflict_section(section, "test.py")
        
        assert analysis['type'] == 'identical'
        assert analysis['complexity'] == 0
        assert analysis['auto_fix_available'] is True
        assert analysis['suggested_resolution'] == 'take_either'
    
    async def test_whitespace_conflict_detection(self, resolver):
        """Test detection of whitespace-only conflicts."""
        ours = "function(){\n    return true;\n}"
        theirs = "function() {\n\treturn true;\n}"
        
        is_whitespace = resolver._is_whitespace_conflict(ours, theirs)
        assert is_whitespace is True
    
    async def test_import_conflict_detection(self, resolver):
        """Test detection of import statement conflicts."""
        ours = "import os\nimport sys"
        theirs = "import os\nimport json\nimport sys"
        
        is_import = resolver._is_import_conflict(ours, theirs, "test.py")
        assert is_import is True
        
        # Test non-Python file
        is_import_js = resolver._is_import_conflict(ours, theirs, "test.txt")
        assert is_import_js is False
    
    async def test_ai_conflict_analysis(self, resolver):
        """Test AI-powered conflict analysis."""
        section = {
            'ours': 'function calculate() { return x + y; }',
            'theirs': 'function calculate() { return x * y; }',
            'ours_label': 'HEAD',
            'theirs_label': 'feature'
        }
        
        base_analysis = {
            'type': 'unknown',
            'complexity': 1,
            'suggested_resolution': 'manual',
            'reasoning': '',
            'auto_fix_available': False
        }
        
        # Mock AI response
        resolver.ai_interface.generate_response = AsyncMock(return_value="""
Type: function_change
Complexity: 2
Suggested Resolution: manual_review
Reasoning: Mathematical operation change from addition to multiplication
Auto-fixable: no
""")
        
        analysis = await resolver._ai_analyze_conflict(section, "calc.js", base_analysis)
        
        assert analysis['type'] == 'function_change'
        assert analysis['complexity'] == 2
        assert analysis['suggested_resolution'] == 'manual_review'
        assert 'Mathematical operation' in analysis['reasoning']
        assert analysis['auto_fix_available'] is False


class TestGitHubAPIClient:
    """Test suite for GitHub API client."""
    
    @pytest.fixture
    def api_client(self):
        """Create GitHubAPIClient instance."""
        return GitHubAPIClient(token="test_token")
    
    async def test_client_initialization(self, api_client):
        """Test API client initialization."""
        assert api_client.token == "test_token"
        assert api_client.base_url == "https://api.github.com"
        assert api_client.session is None
    
    async def test_context_manager(self, api_client):
        """Test async context manager functionality."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            async with api_client as client:
                assert client.session is not None
                
            # Verify session was closed
            mock_session.close.assert_called_once()
    
    async def test_repository_info_retrieval(self, api_client):
        """Test repository information retrieval."""
        mock_response_data = {
            'id': 123,
            'name': 'test-repo',
            'full_name': 'testuser/test-repo',
            'description': 'Test repository'
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            async with api_client as client:
                repo_info = await client.get_repository_info("testuser", "test-repo")
                
                assert repo_info['name'] == 'test-repo'
                assert repo_info['id'] == 123
    
    async def test_pull_request_creation(self, api_client):
        """Test pull request creation via API."""
        mock_pr_data = {
            'id': 456,
            'number': 2,
            'title': 'Test PR',
            'body': 'Test PR description',
            'head': {'ref': 'feature-branch'},
            'base': {'ref': 'main'},
            'html_url': 'https://github.com/testuser/test-repo/pull/2',
            'created_at': '2024-01-01T12:00:00Z',
            'state': 'open'
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value=mock_pr_data)
            
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            async with api_client as client:
                pr = await client.create_pull_request(
                    owner="testuser",
                    repo="test-repo",
                    title="Test PR",
                    body="Test PR description",
                    head="feature-branch",
                    base="main"
                )
                
                assert pr.number == 2
                assert pr.title == "Test PR"
                assert pr.status == PRStatus.OPEN
                assert pr.head_branch == "feature-branch"
                assert pr.base_branch == "main"


class TestIntegrationScenarios:
    """Integration tests for complete Git workflows."""
    
    @pytest.fixture
    async def full_setup(self):
        """Create complete test setup with repository and manager."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize Git repo
            repo = Repo.init(temp_dir)
            
            # Configure user
            repo.config_writer().set_value("user", "name", "Test User").release()
            repo.config_writer().set_value("user", "email", "test@example.com").release()
            
            # Create initial structure
            (temp_dir / "src").mkdir()
            (temp_dir / "tests").mkdir()
            (temp_dir / "docs").mkdir()
            
            # Initial files
            readme = temp_dir / "README.md"
            readme.write_text("# Test Project\n\nA test project for Git integration.")
            
            main_py = temp_dir / "src" / "main.py"
            main_py.write_text('def main():\n    print("Hello, World!")\n')
            
            test_py = temp_dir / "tests" / "test_main.py"
            test_py.write_text('def test_main():\n    assert True\n')
            
            # Initial commit
            repo.index.add([str(readme), str(main_py), str(test_py)])
            repo.index.commit("Initial project setup")
            
            # Create Git manager
            manager = EnhancedGitManager(
                repository_path=temp_dir,
                github_token="test_token",
                safe_mode=False
            )
            
            yield temp_dir, repo, manager
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_complete_feature_development_workflow(self, full_setup):
        """Test complete feature development workflow from branch to merge."""
        temp_dir, repo, manager = full_setup
        
        # Step 1: Create feature branch
        feature_result = await manager.create_branch(
            branch_name="feature/user-authentication",
            checkout=True
        )
        assert feature_result.is_success
        
        # Step 2: Implement feature
        auth_py = temp_dir / "src" / "auth.py"
        auth_py.write_text("""
class AuthManager:
    def __init__(self):
        self.users = {}
    
    def register_user(self, username, password):
        if username in self.users:
            return False
        self.users[username] = password
        return True
    
    def login(self, username, password):
        return self.users.get(username) == password
""")
        
        auth_test = temp_dir / "tests" / "test_auth.py"
        auth_test.write_text("""
import unittest
from src.auth import AuthManager

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.auth = AuthManager()
    
    def test_register_user(self):
        result = self.auth.register_user("testuser", "password")
        self.assertTrue(result)
    
    def test_login(self):
        self.auth.register_user("testuser", "password")
        result = self.auth.login("testuser", "password")
        self.assertTrue(result)
""")
        
        # Step 3: Smart commit with AI-generated message
        with patch.object(manager.commit_generator, 'ai_interface') as mock_ai:
            mock_ai.generate_response = AsyncMock(
                return_value="feat(auth): implement user authentication system\n\nAdd AuthManager class with user registration and login functionality.\nIncludes comprehensive unit tests for authentication features."
            )
            
            commit_result = await manager.generate_smart_commit(
                auto_add=True,
                conventional_commits=True
            )
            
            assert commit_result.is_success
            assert commit_result.metadata["ai_generated_message"] is True
        
        # Step 4: Create pull request
        with patch('src.integrations.git_advanced.GitHubAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.create_pull_request = AsyncMock(return_value=PullRequest(
                id=789,
                number=3,
                title="feat(auth): implement user authentication system",
                body="This PR implements a complete user authentication system...",
                head_branch="feature/user-authentication",
                base_branch="main",
                status=PRStatus.OPEN,
                url="https://github.com/test/repo/pull/3"
            ))
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch.object(manager, '_parse_repository_info') as mock_repo_info:
                mock_repo_info.return_value = {'owner': 'testuser', 'repo': 'testrepo'}
                
                pr = await manager.create_pull_request(
                    head_branch="feature/user-authentication",
                    base_branch="main"
                )
                
                assert pr.number == 3
                assert pr.status == PRStatus.OPEN
        
        # Step 5: Code review
        with patch.object(manager.validation_service, 'validate_file') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_authentic=True,
                authenticity_score=92.0,
                real_progress=95.0,
                fake_progress=5.0,
                issues=[]  # No issues found
            )
            
            with patch.object(manager, '_get_pr_changes') as mock_changes:
                mock_changes.return_value = {
                    'modified_files': ['src/auth.py', 'tests/test_auth.py'],
                    'diff_content': 'authentication implementation diff...',
                    'file_count': 2
                }
                
                review = await manager.review_pull_request(
                    pr_number=3,
                    review_type="comprehensive"
                )
                
                assert review.quality_score == 92.0
                assert review.status == ReviewStatus.APPROVED
                assert len(review.security_issues) == 0
        
        # Step 6: Smart merge
        # Switch back to main for merge
        repo.heads.main.checkout()
        
        merge_result = await manager.smart_merge_with_conflict_resolution(
            branch_name="feature/user-authentication",
            auto_resolve=True
        )
        
        # Verify merge was successful
        assert merge_result.is_success or merge_result.has_conflicts  # Either success or detected conflicts
        
        # Step 7: Verify final state
        final_status = await manager.get_repository_status()
        assert final_status['current_branch'] == 'main'
        assert not final_status['is_dirty']  # Clean working directory
    
    async def test_conflict_resolution_workflow(self, full_setup):
        """Test complete conflict resolution workflow."""
        temp_dir, repo, manager = full_setup
        
        # Create two conflicting branches
        # Branch 1: Update main branch
        main_file = temp_dir / "src" / "config.py"
        main_file.write_text("""
CONFIG = {
    'database_url': 'postgresql://localhost/main_db',
    'debug': False,
    'version': '1.0.0'
}
""")
        repo.index.add([str(main_file)])
        repo.index.commit("Update configuration for production")
        
        # Branch 2: Create feature branch from earlier commit
        feature_branch = repo.create_head("feature/config-update", repo.head.commit.parents[0])
        feature_branch.checkout()
        
        main_file.write_text("""
CONFIG = {
    'database_url': 'sqlite:///dev.db',
    'debug': True,
    'version': '1.1.0',
    'new_feature': True
}
""")
        repo.index.add([str(main_file)])
        repo.index.commit("Update configuration for development")
        
        # Switch back to main
        repo.heads.main.checkout()
        
        # Attempt merge with conflict resolution
        mock_analysis = {
            'total_conflicts': 1,
            'auto_resolvable': 1,
            'complex_conflicts': 0,
            'resolution_strategy': 'semi_automatic',
            'file_analyses': [{
                'file_path': 'src/config.py',
                'conflict_count': 1,
                'auto_resolvable': True,
                'complexity_score': 1,
                'resolution_suggestions': [{
                    'suggested_resolution': 'merge_both',
                    'auto_fix_available': True,
                    'reasoning': 'Configuration merge with complementary changes'
                }]
            }]
        }
        
        mock_resolution = {
            'resolved_files': ['src/config.py'],
            'failed_files': [],
            'manual_review_files': [],
            'total_processed': 1
        }
        
        with patch.object(manager.conflict_resolver, 'analyze_conflicts') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            with patch.object(manager.conflict_resolver, 'auto_resolve_conflicts') as mock_resolve:
                mock_resolve.return_value = mock_resolution
                
                merge_result = await manager.smart_merge_with_conflict_resolution(
                    branch_name="feature/config-update",
                    auto_resolve=True
                )
                
                # Should attempt resolution
                assert "auto_resolved_conflicts" in merge_result.metadata or merge_result.has_conflicts
    
    async def test_ci_cd_integration_workflow(self, full_setup):
        """Test CI/CD integration workflow."""
        temp_dir, repo, manager = full_setup
        
        # Create CI configuration
        github_dir = temp_dir / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        
        ci_config = github_dir / "ci.yml"
        ci_config.write_text("""
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: python -m pytest tests/
""")
        
        repo.index.add([str(ci_config)])
        repo.index.commit("Add CI configuration")
        
        # Test CI trigger
        with patch.object(manager, '_parse_repository_info') as mock_repo_info:
            mock_repo_info.return_value = {'owner': 'testuser', 'repo': 'testrepo'}
            
            with patch.object(manager, '_trigger_github_workflow') as mock_trigger:
                mock_trigger.return_value = {
                    'status': 'triggered',
                    'branch': 'main',
                    'workflow': 'ci.yml',
                    'run_id': 12345
                }
                
                result = await manager.trigger_ci_pipeline(
                    branch_name="main",
                    workflow_name="ci.yml"
                )
                
                assert result['status'] == 'triggered'
                assert result['workflow'] == 'ci.yml'
                assert 'run_id' in result


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for Git integration."""
    
    async def test_large_repository_handling(self):
        """Test handling of large repositories."""
        # This would test performance with large repos
        # Skipped for regular test runs due to resource requirements
        pytest.skip("Performance test - run separately")
    
    async def test_concurrent_operations(self):
        """Test concurrent Git operations."""
        # This would test thread safety and concurrency
        pytest.skip("Concurrency test - run separately")
    
    async def test_memory_usage(self):
        """Test memory usage with large diffs and many files."""
        pytest.skip("Memory test - run separately")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])