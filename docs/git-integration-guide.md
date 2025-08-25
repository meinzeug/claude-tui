# Enhanced Git Integration Guide

## Overview

Claude-TIU's Enhanced Git Integration provides enterprise-grade version control features with AI-powered automation, intelligent code review workflows, and advanced conflict resolution capabilities. This guide covers all aspects of using the enhanced Git features within Claude-TIU.

## Table of Contents

1. [Features Overview](#features-overview)
2. [Getting Started](#getting-started)
3. [Smart Commit Generation](#smart-commit-generation)
4. [Automated Pull Request Management](#automated-pull-request-management)
5. [AI-Powered Code Review](#ai-powered-code-review)
6. [Smart Conflict Resolution](#smart-conflict-resolution)
7. [Branch Strategy Enforcement](#branch-strategy-enforcement)
8. [CI/CD Integration](#cicd-integration)
9. [UI Components](#ui-components)
10. [Configuration](#configuration)
11. [API Reference](#api-reference)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

## Features Overview

### Core Features

- **Smart Commit Generation**: AI-powered commit message generation with conventional commit support
- **Automated Pull Request Management**: Intelligent PR creation with AI-generated titles and descriptions
- **AI-Powered Code Review**: Automated code review with security analysis and quality scoring
- **Smart Conflict Resolution**: AI-assisted merge conflict resolution with auto-resolution capabilities
- **Branch Strategy Enforcement**: Configurable branch protection rules and workflows
- **CI/CD Integration**: Seamless integration with GitHub Actions and other CI/CD platforms
- **Real-time Visualization**: Comprehensive UI for Git workflow management

### Key Benefits

- **Reduced Manual Work**: Automate repetitive Git tasks and workflows
- **Improved Code Quality**: AI-powered code review catches issues early
- **Faster Resolution**: Smart conflict resolution reduces merge time
- **Better Collaboration**: Enhanced PR workflows improve team coordination
- **Quality Assurance**: Integration with anti-hallucination validation ensures code authenticity

## Getting Started

### Installation

The Enhanced Git Integration is included with Claude-TIU. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Setup

1. **Initialize Enhanced Git Manager**:

```python
from src.integrations.git_advanced import EnhancedGitManager

# Initialize with repository path
git_manager = EnhancedGitManager(
    repository_path="/path/to/your/repo",
    github_token="your_github_token",  # Optional
    branch_strategy=BranchStrategy.GITHUB_FLOW
)
```

2. **Configure Authentication**:

```bash
# Set GitHub token (recommended)
export GITHUB_TOKEN="your_personal_access_token"

# Or GitLab token
export GITLAB_TOKEN="your_gitlab_token"
```

3. **Verify Setup**:

```python
# Check repository status
status = await git_manager.get_repository_status()
print(f"Repository: {status['repository_path']}")
print(f"Current branch: {status['current_branch']}")
print(f"Is clean: {not status['is_dirty']}")
```

## Smart Commit Generation

### Overview

Smart Commit Generation uses AI to create meaningful, well-formatted commit messages based on your code changes.

### Basic Usage

```python
# Generate and create a smart commit
result = await git_manager.generate_smart_commit(
    auto_add=True,  # Automatically stage modified files
    conventional_commits=True,  # Use conventional commit format
    validate_message=True  # Validate generated message
)

if result.is_success:
    print(f"Commit created: {result.commit_hash}")
    print(f"Message: {result.message}")
else:
    print(f"Commit failed: {result.message}")
```

### Conventional Commits

When `conventional_commits=True`, the system generates messages following the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Examples:
- `feat(auth): add user authentication system`
- `fix(database): resolve connection pooling issue`
- `docs(readme): update installation instructions`
- `test(api): add comprehensive endpoint tests`

### Commit Types

The AI automatically determines appropriate commit types:

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New features | `feat(api): add user registration endpoint` |
| `fix` | Bug fixes | `fix(auth): handle invalid token gracefully` |
| `docs` | Documentation | `docs(readme): update API documentation` |
| `style` | Code style changes | `style(format): apply consistent indentation` |
| `refactor` | Code refactoring | `refactor(utils): simplify helper functions` |
| `test` | Tests | `test(auth): add unit tests for login flow` |
| `chore` | Maintenance | `chore(deps): update dependencies` |

### Advanced Features

```python
# Custom commit message generator
from src.integrations.git_advanced import CommitMessageGenerator

generator = CommitMessageGenerator(ai_interface=your_ai_interface)

# Generate message for specific changes
message = await generator.generate_commit_message(
    diff_content=repo.git.diff('--cached'),
    file_paths=['src/auth.py', 'tests/test_auth.py'],
    conventional_commits=True
)
```

## Automated Pull Request Management

### Creating Pull Requests

```python
# Create PR with AI-generated content
pr = await git_manager.create_pull_request(
    title=None,  # AI-generated if None
    body=None,   # AI-generated if None
    base_branch="main",
    head_branch=None,  # Current branch if None
    draft=False,
    auto_merge=False
)

print(f"PR #{pr.number} created: {pr.url}")
```

### AI-Generated PR Content

The system automatically generates:

1. **Meaningful Titles**: Based on commit history and changes
2. **Structured Descriptions**: Including:
   - Summary of changes
   - Motivation and context
   - Testing instructions
   - Related issues/tickets

### Example Generated PR

```markdown
# feat(auth): implement user authentication system

## Summary

This PR implements a comprehensive user authentication system with the following features:
- User registration and login
- JWT token management
- Password hashing with bcrypt
- Role-based access control

## Changes Made

- Added `AuthManager` class for user management
- Implemented JWT token generation and validation
- Created authentication middleware for API routes
- Added comprehensive unit tests

## Testing

- Run `pytest tests/test_auth.py` to verify authentication functionality
- Test registration: `POST /api/auth/register`
- Test login: `POST /api/auth/login`

## Related Issues

Closes #123: User Authentication System
```

### Pull Request Status Tracking

```python
# Get PR status
pr_details = await git_manager.get_pull_request(pr_number=1)

print(f"Status: {pr_details.status}")
print(f"Checks passed: {pr_details.checks_passed}")
print(f"Mergeable: {pr_details.mergeable}")
print(f"Conflicts: {len(pr_details.conflicts)}")
```

## AI-Powered Code Review

### Automated Reviews

```python
# Perform comprehensive code review
review = await git_manager.review_pull_request(
    pr_number=1,
    review_type="comprehensive",  # "quick", "standard", "comprehensive"
    auto_approve_safe=True  # Auto-approve if no issues
)

print(f"Review status: {review.status}")
print(f"Quality score: {review.quality_score}/100")
print(f"Issues found: {len(review.security_issues)}")
print(f"Comments: {len(review.comments)}")
```

### Review Types

1. **Quick Review** (< 30 seconds):
   - Basic syntax and style checks
   - Security vulnerability scanning
   - Anti-hallucination validation

2. **Standard Review** (1-2 minutes):
   - Code quality analysis
   - Best practices validation
   - Test coverage assessment
   - Documentation review

3. **Comprehensive Review** (3-5 minutes):
   - Deep code analysis
   - Architecture review
   - Performance considerations
   - Comprehensive security audit

### Review Results

Reviews provide:

```python
# Access review details
print("Security Issues:")
for issue in review.security_issues:
    print(f"  {issue.severity}: {issue.description}")

print("\nSuggestions:")
for suggestion in review.suggestions:
    print(f"  - {suggestion}")

print("\nComments:")
for comment in review.comments:
    print(f"  {comment['path']}:{comment['line']} - {comment['body']}")
```

### Quality Scoring

The AI assigns quality scores based on:
- **Code authenticity** (anti-hallucination validation)
- **Security vulnerabilities** 
- **Best practices compliance**
- **Test coverage and quality**
- **Documentation completeness**
- **Performance considerations**

### Auto-Approval

Safe changes can be automatically approved:

```python
# Configure auto-approval criteria
review = await git_manager.review_pull_request(
    pr_number=1,
    auto_approve_safe=True  # Will approve if:
    # - Quality score > 85
    # - No security issues
    # - No critical or high-severity issues
    # - Passes all validation checks
)
```

## Smart Conflict Resolution

### Conflict Detection

```python
# Detect potential conflicts before merge
conflicts = await git_manager._detect_potential_conflicts("feature/branch")
print(f"Potential conflicts in {len(conflicts)} files")
```

### Automated Resolution

```python
# Attempt smart merge with conflict resolution
result = await git_manager.smart_merge_with_conflict_resolution(
    branch_name="feature/user-auth",
    strategy="adaptive",  # "adaptive", "ours", "theirs", "manual"
    auto_resolve=True,
    validate_merge=True
)

if result.is_success:
    print(f"Merge successful: {result.commit_hash}")
    auto_resolved = result.metadata.get('auto_resolved_conflicts', 0)
    print(f"Auto-resolved {auto_resolved} conflicts")
else:
    print(f"Conflicts remaining: {len(result.conflicts)}")
```

### Conflict Analysis

The AI analyzes conflicts to determine:

```python
# Get detailed conflict analysis
resolver = git_manager.conflict_resolver
analysis = await resolver.analyze_conflicts(repo, conflict_files)

print(f"Total conflicts: {analysis['total_conflicts']}")
print(f"Auto-resolvable: {analysis['auto_resolvable']}")
print(f"Complex conflicts: {analysis['complex_conflicts']}")
print(f"Strategy: {analysis['resolution_strategy']}")
```

### Conflict Types

The system recognizes and handles:

1. **Whitespace Conflicts**: Automatically resolved
2. **Import Conflicts**: Intelligently merged
3. **Simple Additions**: Auto-resolved when non-overlapping
4. **Identical Changes**: Automatically resolved
5. **Complex Logic Changes**: Flagged for manual review

### Manual Resolution Assistance

For complex conflicts, the AI provides:

```python
# Get resolution suggestions
for file_analysis in analysis['file_analyses']:
    for suggestion in file_analysis['resolution_suggestions']:
        print(f"File: {file_analysis['file_path']}")
        print(f"Suggestion: {suggestion['suggested_resolution']}")
        print(f"Reasoning: {suggestion['reasoning']}")
        print(f"Auto-fixable: {suggestion['auto_fix_available']}")
```

## Branch Strategy Enforcement

### Supported Strategies

1. **GitFlow**: Feature branches, develop branch, release branches
2. **GitHub Flow**: Feature branches directly to main
3. **GitLab Flow**: Feature branches with environment branches
4. **Custom**: User-defined branch strategies

### Configuration

```python
from src.integrations.git_advanced import BranchRule, BranchStrategy

# Set branch strategy
git_manager.branch_strategy = BranchStrategy.GITHUB_FLOW

# Define branch protection rules
main_rule = BranchRule(
    pattern="main",
    required_reviews=2,
    dismiss_stale_reviews=True,
    require_code_owner_reviews=True,
    required_status_checks=["ci", "tests", "security-scan"],
    enforce_admins=True,
    allow_force_pushes=False,
    allow_deletions=False
)

# Apply protection
result = await git_manager.enforce_branch_protection("main", main_rule)
```

### Branch Rules

Create sophisticated protection rules:

```python
# Feature branch rules
feature_rule = BranchRule(
    pattern="feature/*",
    required_reviews=1,
    required_status_checks=["tests"],
    allow_force_pushes=True,  # Allow during development
    allow_deletions=True      # Allow cleanup after merge
)

# Development branch rules
develop_rule = BranchRule(
    pattern="develop",
    required_reviews=1,
    dismiss_stale_reviews=True,
    required_status_checks=["ci", "integration-tests"],
    allow_force_pushes=False
)

# Apply all rules
for pattern, rule in [("main", main_rule), ("feature/*", feature_rule), ("develop", develop_rule)]:
    await git_manager.enforce_branch_protection(pattern, rule)
```

### Validation

Branch operations are validated against rules:

```python
# Branch creation validation
try:
    result = await git_manager.create_branch("hotfix/critical-fix")
    # Will validate against strategy and rules
except Exception as e:
    print(f"Branch creation blocked: {e}")
```

## CI/CD Integration

### GitHub Actions Integration

```python
# Trigger CI pipeline
result = await git_manager.trigger_ci_pipeline(
    branch_name="feature/new-api",
    workflow_name="ci.yml",
    inputs={
        "environment": "staging",
        "run_tests": True,
        "deploy": False
    }
)

print(f"Pipeline status: {result['status']}")
print(f"Run ID: {result.get('run_id')}")
```

### Workflow Configuration

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src/
      - name: Run security scan
        run: |
          bandit -r src/
```

### Status Checks

Configure required status checks:

```python
# Configure CI integration
git_manager.branch_rules["main"].required_status_checks = [
    "ci",
    "tests",
    "security-scan",
    "code-quality"
]
```

### Local Hooks

For repositories without CI/CD platforms:

```python
# Trigger local CI hooks
result = await git_manager._trigger_local_ci_hooks(
    branch_name="feature/branch",
    inputs={"environment": "test"}
)
```

## UI Components

### Git Workflow Widget

The main UI component provides comprehensive Git management:

```python
from src.ui.widgets.git_workflow_widget import GitWorkflowWidget

# Initialize widget
git_widget = GitWorkflowWidget(repository_path=Path("/path/to/repo"))

# Widget includes tabs for:
# - Repository Status
# - Branch Management  
# - Pull Requests
# - Code Review
# - Conflict Resolution
```

### Individual Components

```python
from src.ui.widgets.git_workflow_widget import (
    GitStatusWidget,
    BranchTreeWidget,
    PullRequestDashboard,
    CodeReviewWidget,
    ConflictResolutionWidget
)

# Use individual components
status_widget = GitStatusWidget(git_manager)
branch_widget = BranchTreeWidget(git_manager)
pr_dashboard = PullRequestDashboard(git_manager)
```

### Key Bindings

The Git Workflow Widget includes keyboard shortcuts:

- `r`: Refresh current view
- `c`: Create smart commit
- `p`: Create pull request  
- `m`: Smart merge
- `q`: Quit

### Real-time Updates

Components automatically refresh:

```python
# Status widget updates every 10 seconds
# Pull request dashboard monitors PR status
# Conflict resolution widget tracks merge state
```

## Configuration

### Environment Variables

```bash
# GitHub integration
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
export GITHUB_USERNAME="your_username"

# GitLab integration  
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"
export GITLAB_URL="https://gitlab.com"

# AI configuration
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
export AI_MODEL="gpt-4"

# Git configuration
export GIT_DEFAULT_BRANCH="main"
export GIT_AUTHOR_NAME="Your Name"
export GIT_AUTHOR_EMAIL="your.email@example.com"
```

### Configuration File

Create `config/git_config.yaml`:

```yaml
git:
  default_branch: "main"
  branch_strategy: "github_flow"
  
  # Commit configuration
  conventional_commits: true
  auto_add_modified: true
  validate_messages: true
  
  # PR configuration
  auto_review: true
  auto_approve_safe: true
  draft_by_default: false
  
  # Conflict resolution
  auto_resolve_simple: true
  max_auto_resolve_complexity: 2
  
  # Branch protection
  enforce_protection: true
  required_reviews:
    main: 2
    develop: 1
    "feature/*": 1
  
  # CI/CD
  trigger_on_push: true
  required_checks:
    - "ci"
    - "tests" 
    - "security"

ai:
  provider: "openai"
  model: "gpt-4"
  max_tokens: 500
  temperature: 0.3

github:
  api_url: "https://api.github.com"
  timeout: 30
  retry_attempts: 3
```

### Programmatic Configuration

```python
# Configure Git manager
git_manager = EnhancedGitManager(
    repository_path="/path/to/repo",
    github_token=os.getenv("GITHUB_TOKEN"),
    branch_strategy=BranchStrategy.GITHUB_FLOW,
    auto_stage=True,
    safe_mode=True,
    max_file_size=10 * 1024 * 1024  # 10MB
)

# Configure AI interface
from src.core.ai_interface import AIInterface

ai_interface = AIInterface(
    provider="openai",
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=500,
    temperature=0.3
)

git_manager.commit_generator.ai_interface = ai_interface
git_manager.conflict_resolver.ai_interface = ai_interface
```

## API Reference

### EnhancedGitManager

Main class for Git operations with AI enhancement.

#### Methods

##### `generate_smart_commit(auto_add=True, conventional_commits=True, validate_message=True)`

Generate and create AI-powered commit.

**Parameters:**
- `auto_add` (bool): Automatically stage modified files
- `conventional_commits` (bool): Use conventional commit format
- `validate_message` (bool): Validate generated message

**Returns:** `GitOperationResult`

##### `create_pull_request(title=None, body=None, base_branch="main", head_branch=None, draft=False, auto_merge=False)`

Create pull request with AI-generated content.

**Parameters:**
- `title` (str, optional): PR title (AI-generated if None)
- `body` (str, optional): PR description (AI-generated if None)
- `base_branch` (str): Target branch
- `head_branch` (str, optional): Source branch
- `draft` (bool): Create as draft
- `auto_merge` (bool): Enable auto-merge

**Returns:** `PullRequest`

##### `review_pull_request(pr_number, review_type="standard", auto_approve_safe=False)`

Perform AI-powered code review.

**Parameters:**
- `pr_number` (int): Pull request number
- `review_type` (str): "quick", "standard", or "comprehensive"
- `auto_approve_safe` (bool): Auto-approve safe changes

**Returns:** `CodeReview`

##### `smart_merge_with_conflict_resolution(branch_name, strategy="adaptive", auto_resolve=True, validate_merge=True)`

Smart merge with AI conflict resolution.

**Parameters:**
- `branch_name` (str): Branch to merge
- `strategy` (str): Merge strategy
- `auto_resolve` (bool): Attempt auto-resolution
- `validate_merge` (bool): Validate merge result

**Returns:** `GitOperationResult`

##### `enforce_branch_protection(branch_pattern, rule=None)`

Enforce branch protection rules.

**Parameters:**
- `branch_pattern` (str): Branch pattern to protect
- `rule` (BranchRule, optional): Protection rule

**Returns:** `GitOperationResult`

##### `trigger_ci_pipeline(branch_name=None, workflow_name=None, inputs=None)`

Trigger CI/CD pipeline.

**Parameters:**
- `branch_name` (str, optional): Target branch
- `workflow_name` (str, optional): Workflow to trigger
- `inputs` (dict, optional): Workflow inputs

**Returns:** `dict`

### Data Models

#### `PullRequest`

```python
@dataclass
class PullRequest:
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
```

#### `CodeReview`

```python
@dataclass
class CodeReview:
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
```

#### `BranchRule`

```python
@dataclass
class BranchRule:
    pattern: str
    required_reviews: int = 1
    dismiss_stale_reviews: bool = True
    require_code_owner_reviews: bool = False
    required_status_checks: List[str] = field(default_factory=list)
    enforce_admins: bool = True
    allow_force_pushes: bool = False
    allow_deletions: bool = False
```

### Enums

#### `PRStatus`

- `DRAFT`: Draft pull request
- `OPEN`: Open pull request
- `CLOSED`: Closed pull request
- `MERGED`: Merged pull request
- `CONFLICTED`: Pull request with conflicts

#### `ReviewStatus`

- `PENDING`: Review pending
- `APPROVED`: Changes approved
- `CHANGES_REQUESTED`: Changes requested
- `COMMENTED`: Comments only

#### `CIStatus`

- `PENDING`: CI pending
- `RUNNING`: CI running
- `SUCCESS`: CI successful
- `FAILURE`: CI failed
- `CANCELLED`: CI cancelled

## Best Practices

### Commit Messages

1. **Use Conventional Commits**: Enable structured commit history
2. **Be Descriptive**: Let AI generate detailed, meaningful messages
3. **Include Context**: AI considers file changes and project context
4. **Validate Messages**: Always validate generated messages

```python
# Best practice example
result = await git_manager.generate_smart_commit(
    auto_add=True,
    conventional_commits=True,
    validate_message=True
)
```

### Pull Requests

1. **Meaningful Titles**: Let AI generate descriptive titles
2. **Structured Descriptions**: Use AI-generated structured content
3. **Regular Reviews**: Enable automated code reviews
4. **Status Checks**: Require CI/CD validation

```python
# Best practice PR creation
pr = await git_manager.create_pull_request(
    base_branch="main",
    draft=False,  # Ready for review
    auto_merge=False  # Manual merge after approval
)

# Immediate review
review = await git_manager.review_pull_request(
    pr_number=pr.number,
    review_type="comprehensive",
    auto_approve_safe=True
)
```

### Branch Management

1. **Consistent Strategy**: Choose and stick to one strategy
2. **Protection Rules**: Implement appropriate branch protection
3. **Regular Cleanup**: Remove merged feature branches
4. **Naming Conventions**: Use consistent branch naming

```python
# Best practice branch protection
rules = {
    "main": BranchRule(
        pattern="main",
        required_reviews=2,
        required_status_checks=["ci", "tests", "security"],
        enforce_admins=True,
        allow_force_pushes=False
    ),
    "feature/*": BranchRule(
        pattern="feature/*",
        required_reviews=1,
        allow_force_pushes=True,
        allow_deletions=True
    )
}

for pattern, rule in rules.items():
    await git_manager.enforce_branch_protection(pattern, rule)
```

### Conflict Resolution

1. **Auto-resolve Simple**: Enable automatic resolution for simple conflicts
2. **Review Complex**: Always review AI suggestions for complex conflicts
3. **Validate Results**: Validate merge results after resolution
4. **Test Thoroughly**: Run comprehensive tests after conflict resolution

```python
# Best practice conflict resolution
result = await git_manager.smart_merge_with_conflict_resolution(
    branch_name="feature/branch",
    strategy="adaptive",  # Let AI choose best strategy
    auto_resolve=True,    # Auto-resolve simple conflicts
    validate_merge=True   # Validate result
)

# Always check results
if result.has_conflicts:
    print(f"Manual resolution needed for: {result.conflicts}")
elif result.is_success:
    print("Merge successful with validation")
```

### Code Review

1. **Appropriate Depth**: Choose review type based on change complexity
2. **Address Issues**: Always address security and critical issues
3. **Quality Gates**: Set minimum quality thresholds
4. **Continuous Learning**: Review AI suggestions to improve over time

```python
# Best practice code review
review = await git_manager.review_pull_request(
    pr_number=pr_number,
    review_type="comprehensive" if is_critical_change else "standard",
    auto_approve_safe=False  # Manual approval for important changes
)

# Quality gate
if review.quality_score < 80 or review.security_issues:
    print("Quality gate failed - address issues before merge")
    return False
```

### CI/CD Integration

1. **Required Checks**: Make CI checks required for merge
2. **Fast Feedback**: Configure fast-failing tests first
3. **Security Scans**: Include security scanning in pipeline
4. **Deployment Gates**: Use quality gates for deployment

```yaml
# Best practice GitHub Actions workflow
name: Quality Gate
on:
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - name: Code Review
        run: claude-tiu git review --pr ${{ github.event.number }}
      - name: Security Scan
        run: bandit -r src/
      - name: Tests
        run: pytest tests/ --cov=80
      - name: Quality Gate
        run: |
          if [ "$QUALITY_SCORE" -lt 85 ]; then
            echo "Quality gate failed"
            exit 1
          fi
```

## Troubleshooting

### Common Issues

#### Authentication Errors

**Problem**: "GitHub API authentication failed"

**Solution**:
1. Verify GitHub token is set: `echo $GITHUB_TOKEN`
2. Check token permissions (repo, workflow, admin:repo_hook)
3. Regenerate token if expired

```bash
# Set token correctly
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"

# Test authentication
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

#### AI Service Errors

**Problem**: "AI commit message generation failed"

**Solution**:
1. Verify OpenAI API key: `echo $OPENAI_API_KEY`
2. Check API quota and usage
3. Fallback to manual commit messages

```python
# Fallback on AI failure
try:
    result = await git_manager.generate_smart_commit()
except Exception as e:
    print(f"AI failed: {e}")
    # Use traditional commit
    result = await git_manager.commit(
        message="Manual commit message",
        auto_add=True
    )
```

#### Merge Conflicts

**Problem**: "Auto-resolution failed for complex conflicts"

**Solution**:
1. Review conflict analysis
2. Use manual resolution for complex cases
3. Validate results after resolution

```python
# Handle complex conflicts
analysis = await git_manager.conflict_resolver.analyze_conflicts(repo, conflicts)

if analysis['resolution_strategy'] == 'manual':
    print("Complex conflicts require manual resolution")
    # Provide detailed guidance
    for file_analysis in analysis['file_analyses']:
        print(f"File: {file_analysis['file_path']}")
        for suggestion in file_analysis['resolution_suggestions']:
            print(f"  - {suggestion['reasoning']}")
```

#### Performance Issues

**Problem**: "Git operations are slow"

**Solutions**:
1. Enable shallow clones for large repositories
2. Configure Git LFS for large files
3. Optimize AI model usage

```python
# Performance optimization
git_manager = EnhancedGitManager(
    repository_path=repo_path,
    max_file_size=5 * 1024 * 1024,  # Reduce file size limit
    safe_mode=False  # Disable for performance (use carefully)
)

# Use quick reviews for non-critical changes
review = await git_manager.review_pull_request(
    pr_number=pr_number,
    review_type="quick"  # Faster than comprehensive
)
```

### Debug Mode

Enable detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src.integrations.git_advanced')
logger.setLevel(logging.DEBUG)

# Git operations will now provide detailed logs
```

### Health Checks

Verify system health:

```python
# Check Git manager health
health = await git_manager.health_check()

print(f"Status: {health['status']}")
print(f"Git version: {health['git_version']}")
print(f"Repository integrity: {health['integrity_ok']}")
print(f"Metrics: {health['metrics']}")
```

### Support and Resources

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/claude-tiu/issues)
- **Documentation**: [Complete API documentation](https://docs.claude-tiu.org)
- **Examples**: [Sample implementations](https://github.com/your-org/claude-tiu/tree/main/examples)
- **Community**: [Join our Discord](https://discord.gg/claude-tiu)

---

## Conclusion

The Enhanced Git Integration transforms version control workflows with AI-powered automation while maintaining enterprise-grade security and reliability. By combining intelligent automation with comprehensive validation, teams can achieve higher productivity without sacrificing code quality.

For advanced use cases and custom implementations, refer to the API documentation and explore the extensive configuration options available.