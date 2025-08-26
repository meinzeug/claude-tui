"""
Claude Code and Claude Flow Integration Tests.

Tests the authentication system integration with:
- Claude Code authentication flows
- Claude Flow OAuth and session management
- API token validation
- Cross-service communication
- Security compliance for Claude services
"""

import pytest
import asyncio
import uuid
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import authentication and Claude integration components
sys.path.append('/home/tekkadmin/claude-tui/src')
from auth.auth_service import AuthenticationService
from auth.security_compliance import run_security_compliance_check
from core.exceptions import AuthenticationError


class TestClaudeCodeIntegration:
    """Test Claude Code authentication integration."""
    
    @pytest.fixture
    def claude_auth_service(self):
        """Create authentication service configured for Claude Code."""
        return AuthenticationService(
            secret_key="claude-code-secret-key-2024",
            oauth_config={
                "github": {
                    "client_id": "claude-code-github-client",
                    "client_secret": "claude-code-github-secret",
                    "redirect_uri": "https://claude-code.anthropic.com/auth/callback"
                }
            }
        )
    
    @pytest.fixture
    def claude_user(self):
        """Create Claude Code user for testing."""
        user = Mock()
        user.id = uuid.uuid4()
        user.username = "claude_developer"
        user.email = "dev@anthropic.com"
        user.full_name = "Claude Developer"
        user.is_active = True
        user.is_verified = True
        user.role = "developer"
        return user
    
    @pytest.mark.asyncio
    async def test_claude_code_authentication(self, claude_auth_service, claude_user):
        """Test Claude Code user authentication flow."""
        # Mock session and token creation
        mock_session = Mock()
        mock_session.session_id = "claude-code-session-123"
        mock_session.expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
        
        mock_tokens = Mock()
        mock_tokens.access_token = "claude-code-access-token"
        mock_tokens.refresh_token = "claude-code-refresh-token"
        mock_tokens.token_type = "bearer"
        mock_tokens.expires_in = 3600
        
        with patch.object(claude_auth_service.user_service, 'authenticate_user', return_value=claude_user):
            with patch.object(claude_auth_service.session_manager, 'create_session', return_value=mock_session):
                with patch.object(claude_auth_service.user_service, 'get_user_permissions', return_value=['claude:code', 'read', 'write']):
                    with patch.object(claude_auth_service.jwt_service, 'create_token_pair', return_value=mock_tokens):
                        with patch.object(claude_auth_service.audit_logger, 'log_authentication'):
                            result = await claude_auth_service.authenticate_user(
                                identifier="dev@anthropic.com",
                                password="claude-secure-password",
                                ip_address="127.0.0.1",
                                user_agent="Claude-Code/3.0"
                            )
                            
                            # Verify Claude Code specific assertions
                            assert result['user']['username'] == "claude_developer"
                            assert result['user']['email'] == "dev@anthropic.com"
                            assert 'claude:code' in result['user']['permissions']
                            assert result['tokens']['access_token'] == "claude-code-access-token"
                            assert result['session']['session_id'] == "claude-code-session-123"
    
    @pytest.mark.asyncio
    async def test_claude_code_api_token_validation(self, claude_auth_service, claude_user):
        """Test Claude Code API token validation."""
        # Mock token validation for Claude Code API calls
        mock_token_data = Mock()
        mock_token_data.user_id = str(claude_user.id)
        mock_token_data.username = claude_user.username
        mock_token_data.permissions = ['claude:code', 'api:access']
        mock_token_data.session_id = "claude-session-456"
        
        mock_session = Mock()
        mock_session.dict.return_value = {
            'session_id': 'claude-session-456',
            'user_id': str(claude_user.id),
            'ip_address': '127.0.0.1',
            'user_agent': 'Claude-Code/3.0'
        }
        
        with patch.object(claude_auth_service.jwt_service, 'validate_access_token', return_value=mock_token_data):
            with patch.object(claude_auth_service.user_service, 'get_user_by_id', return_value=claude_user):
                with patch.object(claude_auth_service.session_manager, 'get_session', return_value=mock_session):
                    result = await claude_auth_service.validate_token("claude-api-token")
                    
                    assert result['valid'] is True
                    assert result['user']['username'] == "claude_developer"
                    assert 'claude:code' in result['token_data'].permissions
                    assert result['session']['session_id'] == 'claude-session-456'
    
    @pytest.mark.asyncio
    async def test_claude_code_session_management(self, claude_auth_service):
        """Test Claude Code session management features."""
        user_id = str(uuid.uuid4())
        
        # Mock multiple Claude Code sessions
        mock_sessions = [
            Mock(**{
                'dict.return_value': {
                    'session_id': 'claude-desktop-session',
                    'user_id': user_id,
                    'user_agent': 'Claude-Code-Desktop/3.0',
                    'ip_address': '192.168.1.100',
                    'created_at': '2024-01-01T10:00:00Z'
                }
            }),
            Mock(**{
                'dict.return_value': {
                    'session_id': 'claude-web-session',
                    'user_id': user_id,
                    'user_agent': 'Claude-Code-Web/3.0',
                    'ip_address': '192.168.1.101',
                    'created_at': '2024-01-01T11:00:00Z'
                }
            })
        ]
        
        with patch.object(claude_auth_service.session_manager, 'get_user_sessions', return_value=mock_sessions):
            sessions = await claude_auth_service.get_user_sessions(user_id)
            
            assert len(sessions) == 2
            assert any('Claude-Code-Desktop' in session['user_agent'] for session in sessions)
            assert any('Claude-Code-Web' in session['user_agent'] for session in sessions)
    
    @pytest.mark.asyncio
    async def test_claude_code_permission_validation(self, claude_auth_service, claude_user):
        """Test Claude Code specific permission validation."""
        claude_permissions = [
            'claude:code',
            'claude:read',
            'claude:write',
            'claude:execute',
            'files:read',
            'files:write',
            'terminal:access'
        ]
        
        with patch.object(claude_auth_service.user_service, 'get_user_permissions', return_value=claude_permissions):
            permissions = await claude_auth_service.user_service.get_user_permissions(str(claude_user.id))
            
            # Check Claude Code specific permissions
            assert 'claude:code' in permissions
            assert 'claude:execute' in permissions
            assert 'files:read' in permissions
            assert 'terminal:access' in permissions


class TestClaudeFlowIntegration:
    """Test Claude Flow OAuth and workflow integration."""
    
    @pytest.fixture
    def claude_flow_service(self):
        """Create authentication service configured for Claude Flow."""
        return AuthenticationService(
            secret_key="claude-flow-secret-key-2024",
            oauth_config={
                "github": {
                    "client_id": "claude-flow-github-client",
                    "client_secret": "claude-flow-github-secret",
                    "redirect_uri": "https://claude-flow.anthropic.com/auth/callback"
                },
                "google": {
                    "client_id": "claude-flow-google-client",
                    "client_secret": "claude-flow-google-secret",
                    "redirect_uri": "https://claude-flow.anthropic.com/auth/google/callback"
                }
            }
        )
    
    @pytest.fixture
    def claude_flow_user(self):
        """Create Claude Flow user for testing."""
        user = Mock()
        user.id = uuid.uuid4()
        user.username = "flow_orchestrator"
        user.email = "flow@anthropic.com"
        user.full_name = "Claude Flow Orchestrator"
        user.is_active = True
        user.is_verified = True
        user.role = "orchestrator"
        return user
    
    def test_claude_flow_oauth_url_generation(self, claude_flow_service):
        """Test Claude Flow OAuth URL generation for multiple providers."""
        # Test GitHub OAuth URL
        github_result = claude_flow_service.get_oauth_authorization_url(
            provider="github",
            redirect_after_auth="/dashboard/workflows"
        )
        
        assert "authorization_url" in github_result
        assert "github.com" in github_result["authorization_url"]
        assert github_result["provider"] == "github"
        assert "claude-flow-github-client" in github_result["authorization_url"]
        
        # Test Google OAuth URL
        google_result = claude_flow_service.get_oauth_authorization_url(
            provider="google",
            redirect_after_auth="/dashboard/agents"
        )
        
        assert "authorization_url" in google_result
        assert "accounts.google.com" in google_result["authorization_url"]
        assert google_result["provider"] == "google"
    
    @pytest.mark.asyncio
    async def test_claude_flow_github_oauth_callback(self, claude_flow_service, claude_flow_user):
        """Test Claude Flow GitHub OAuth callback handling."""
        # Generate OAuth state first
        oauth_url_result = claude_flow_service.get_oauth_authorization_url("github")
        state = oauth_url_result["state"]
        
        # Mock OAuth callback result
        mock_oauth_result = {
            'user': {
                'id': str(claude_flow_user.id),
                'username': 'flow_orchestrator',
                'email': 'flow@anthropic.com',
                'permissions': ['claude:flow', 'github:read', 'workflows:execute']
            },
            'tokens': {
                'access_token': 'claude-flow-github-token',
                'refresh_token': 'claude-flow-github-refresh'
            },
            'session': {
                'session_id': 'claude-flow-github-session',
                'expires_at': (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            },
            'oauth': {
                'provider': 'github',
                'provider_user_info': {
                    'login': 'anthropic-flow',
                    'name': 'Claude Flow Bot',
                    'email': 'flow@anthropic.com'
                }
            },
            'redirect_after_auth': None
        }
        
        with patch.object(claude_flow_service.oauth_manager, 'handle_callback', return_value=mock_oauth_result):
            with patch.object(claude_flow_service.audit_logger, 'log_authentication'):
                result = await claude_flow_service.handle_oauth_callback(
                    provider="github",
                    code="github-oauth-code-123",
                    state=state,
                    ip_address="127.0.0.1",
                    user_agent="Claude-Flow/2.0"
                )
                
                assert result['user']['username'] == 'flow_orchestrator'
                assert 'claude:flow' in result['user']['permissions']
                assert result['oauth']['provider'] == 'github'
                assert result['tokens']['access_token'] == 'claude-flow-github-token'
    
    @pytest.mark.asyncio
    async def test_claude_flow_workflow_authentication(self, claude_flow_service, claude_flow_user):
        """Test authentication for Claude Flow workflow execution."""
        # Mock workflow execution context
        workflow_context = {
            'workflow_id': 'claude-workflow-123',
            'execution_id': 'exec-456',
            'required_permissions': ['claude:flow', 'workflows:execute', 'agents:spawn']
        }
        
        mock_token_data = Mock()
        mock_token_data.user_id = str(claude_flow_user.id)
        mock_token_data.username = claude_flow_user.username
        mock_token_data.permissions = ['claude:flow', 'workflows:execute', 'agents:spawn']
        mock_token_data.session_id = "workflow-session-789"
        
        with patch.object(claude_flow_service.jwt_service, 'validate_access_token', return_value=mock_token_data):
            with patch.object(claude_flow_service.user_service, 'get_user_by_id', return_value=claude_flow_user):
                with patch.object(claude_flow_service.session_manager, 'get_session'):
                    result = await claude_flow_service.validate_token("workflow-execution-token")
                    
                    # Verify workflow permissions
                    user_permissions = result['token_data'].permissions
                    for required_permission in workflow_context['required_permissions']:
                        assert required_permission in user_permissions
    
    @pytest.mark.asyncio
    async def test_claude_flow_multi_agent_session(self, claude_flow_service):
        """Test Claude Flow multi-agent session management."""
        user_id = str(uuid.uuid4())
        
        # Mock multiple agent sessions
        mock_agent_sessions = [
            Mock(**{
                'dict.return_value': {
                    'session_id': 'agent-researcher-session',
                    'user_id': user_id,
                    'metadata': {'agent_type': 'researcher', 'task_id': 'research-001'},
                    'user_agent': 'Claude-Flow-Agent/2.0',
                    'created_at': '2024-01-01T12:00:00Z'
                }
            }),
            Mock(**{
                'dict.return_value': {
                    'session_id': 'agent-coder-session',
                    'user_id': user_id,
                    'metadata': {'agent_type': 'coder', 'task_id': 'code-001'},
                    'user_agent': 'Claude-Flow-Agent/2.0',
                    'created_at': '2024-01-01T12:05:00Z'
                }
            }),
            Mock(**{
                'dict.return_value': {
                    'session_id': 'orchestrator-session',
                    'user_id': user_id,
                    'metadata': {'agent_type': 'orchestrator', 'workflow_id': 'wf-001'},
                    'user_agent': 'Claude-Flow-Orchestrator/2.0',
                    'created_at': '2024-01-01T12:10:00Z'
                }
            })
        ]
        
        with patch.object(claude_flow_service.session_manager, 'get_user_sessions', return_value=mock_agent_sessions):
            sessions = await claude_flow_service.get_user_sessions(user_id)
            
            assert len(sessions) == 3
            agent_types = [session.get('metadata', {}).get('agent_type') for session in sessions]
            assert 'researcher' in agent_types
            assert 'coder' in agent_types
            assert 'orchestrator' in agent_types


class TestClaudeSecurityCompliance:
    """Test security compliance for Claude services."""
    
    def test_claude_code_security_configuration(self):
        """Test Claude Code security configuration compliance."""
        claude_code_config = {
            'password_policy': {
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_digits': True,
                'require_special': True,
                'max_age_days': 0  # No forced expiration
            },
            'jwt_config': {
                'algorithm': 'HS256',
                'secret_key': 'claude-code-ultra-secure-secret-key-256-bits-entropy-2024',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7,
                'issuer': 'claude-code.anthropic.com',
                'audience': 'claude-code-api'
            },
            'session_config': {
                'session_timeout_minutes': 480,  # 8 hours
                'max_sessions_per_user': 5,
                'secure_cookies': True,
                'httponly_cookies': True,
                'samesite_cookies': 'Strict',
                'session_regeneration': True
            },
            'oauth_config': {
                'providers': {
                    'github': {
                        'client_id': 'claude-code-github-client-id',
                        'client_secret': 'claude-code-github-secret-32-chars-plus',
                        'redirect_uri': 'https://claude-code.anthropic.com/auth/callback'
                    }
                },
                'state_validation': True,
                'pkce_enabled': True,
                'redirect_uri_validation': True
            },
            'encryption_config': {
                'password_hash_algorithm': 'bcrypt',
                'hash_rounds': 12,
                'encryption_key_length': 256,
                'tls_version': '1.3'
            },
            'audit_config': {
                'enabled': True,
                'log_authentication': True,
                'log_authorization': True,
                'log_password_changes': True,
                'log_failed_attempts': True,
                'structured_logging': True,
                'log_retention_days': 90
            },
            'rate_limit_config': {
                'enabled': True,
                'login_attempts_limit': 5,
                'login_window_minutes': 15,
                'api_requests_limit': 1000,
                'api_window_minutes': 60,
                'account_lockout_enabled': True,
                'lockout_duration_minutes': 30
            }
        }
        
        compliance_report = run_security_compliance_check(claude_code_config)
        
        # Verify high compliance for Claude Code
        assert compliance_report['summary']['success_rate'] >= 90
        assert compliance_report['summary']['risk_score'] <= 20
        assert compliance_report['summary']['compliance_level'] in ['low', 'medium']
        
        # Check specific framework compliance
        assert compliance_report['compliance_frameworks']['owasp_top_10']['compliance_percentage'] >= 80
        assert compliance_report['compliance_frameworks']['nist_framework']['compliance_percentage'] >= 80
    
    def test_claude_flow_security_configuration(self):
        """Test Claude Flow security configuration compliance."""
        claude_flow_config = {
            'password_policy': {
                'min_length': 14,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_digits': True,
                'require_special': True,
                'max_age_days': 0
            },
            'jwt_config': {
                'algorithm': 'HS384',
                'secret_key': 'claude-flow-ultra-secure-secret-key-384-bits-entropy-maximum-security-2024',
                'access_token_expire_minutes': 15,  # Shorter for flow operations
                'refresh_token_expire_days': 1,     # Shorter for security
                'issuer': 'claude-flow.anthropic.com',
                'audience': 'claude-flow-api'
            },
            'session_config': {
                'session_timeout_minutes': 240,  # 4 hours for active workflows
                'max_sessions_per_user': 10,     # More sessions for multiple agents
                'secure_cookies': True,
                'httponly_cookies': True,
                'samesite_cookies': 'Strict',
                'session_regeneration': True
            },
            'oauth_config': {
                'providers': {
                    'github': {
                        'client_id': 'claude-flow-github-client-id',
                        'client_secret': 'claude-flow-github-secret-very-long-and-secure',
                        'redirect_uri': 'https://claude-flow.anthropic.com/auth/callback'
                    },
                    'google': {
                        'client_id': 'claude-flow-google-client-id',
                        'client_secret': 'claude-flow-google-secret-very-long-and-secure',
                        'redirect_uri': 'https://claude-flow.anthropic.com/auth/google/callback'
                    }
                },
                'state_validation': True,
                'pkce_enabled': True,
                'redirect_uri_validation': True
            },
            'encryption_config': {
                'password_hash_algorithm': 'bcrypt',
                'hash_rounds': 14,  # Higher for flow security
                'encryption_key_length': 256,
                'tls_version': '1.3'
            },
            'audit_config': {
                'enabled': True,
                'log_authentication': True,
                'log_authorization': True,
                'log_password_changes': True,
                'log_failed_attempts': True,
                'structured_logging': True,
                'log_retention_days': 365  # Longer retention for workflow analysis
            },
            'rate_limit_config': {
                'enabled': True,
                'login_attempts_limit': 3,  # Stricter for flow
                'login_window_minutes': 10,
                'api_requests_limit': 5000,  # Higher for agent operations
                'api_window_minutes': 60,
                'account_lockout_enabled': True,
                'lockout_duration_minutes': 60
            }
        }
        
        compliance_report = run_security_compliance_check(claude_flow_config)
        
        # Verify very high compliance for Claude Flow
        assert compliance_report['summary']['success_rate'] >= 95
        assert compliance_report['summary']['risk_score'] <= 15
        assert compliance_report['summary']['compliance_level'] == 'low'
        
        # Check framework compliance
        frameworks = compliance_report['compliance_frameworks']
        assert frameworks['owasp_top_10']['compliance_percentage'] >= 90
        assert frameworks['nist_framework']['compliance_percentage'] >= 85
        assert frameworks['iso_27001']['compliance_percentage'] >= 85


class TestCrossServiceAuthentication:
    """Test authentication between Claude Code and Claude Flow."""
    
    @pytest.mark.asyncio
    async def test_claude_code_to_flow_token_exchange(self):
        """Test token exchange from Claude Code to Claude Flow."""
        # Simulate Claude Code user wanting to access Claude Flow
        code_auth_service = AuthenticationService(secret_key="claude-code-secret")
        flow_auth_service = AuthenticationService(secret_key="claude-flow-secret")
        
        # Mock Claude Code token validation
        mock_code_token_data = Mock()
        mock_code_token_data.user_id = "claude-user-123"
        mock_code_token_data.username = "claude_developer"
        mock_code_token_data.permissions = ['claude:code', 'claude:flow']
        
        mock_user = Mock()
        mock_user.id = uuid.UUID(mock_code_token_data.user_id)
        mock_user.username = mock_code_token_data.username
        mock_user.email = "dev@anthropic.com"
        mock_user.is_active = True
        
        # Mock Flow session and token creation
        mock_flow_session = Mock()
        mock_flow_session.session_id = "cross-service-session"
        mock_flow_session.expires_at = datetime.now(timezone.utc) + timedelta(hours=4)
        
        mock_flow_tokens = Mock()
        mock_flow_tokens.access_token = "claude-flow-cross-service-token"
        mock_flow_tokens.refresh_token = "claude-flow-cross-service-refresh"
        mock_flow_tokens.token_type = "bearer"
        mock_flow_tokens.expires_in = 900  # 15 minutes
        
        # Test cross-service authentication
        with patch.object(code_auth_service.jwt_service, 'validate_access_token', return_value=mock_code_token_data):
            with patch.object(flow_auth_service.user_service, 'get_user_by_id', return_value=mock_user):
                with patch.object(flow_auth_service.session_manager, 'create_session', return_value=mock_flow_session):
                    with patch.object(flow_auth_service.jwt_service, 'create_token_pair', return_value=mock_flow_tokens):
                        with patch.object(flow_auth_service.audit_logger, 'log_authentication'):
                            # Validate Code token first
                            code_validation = await code_auth_service.validate_token("claude-code-token")
                            assert code_validation['valid'] is True
                            assert 'claude:flow' in code_validation['token_data'].permissions
                            
                            # Create Flow session for validated user
                            flow_result = await flow_auth_service.authenticate_user(
                                identifier=mock_user.email,
                                password="cross-service-auth",  # Special cross-service password
                                ip_address="127.0.0.1",
                                user_agent="Claude-Code-to-Flow/1.0"
                            )
                            
                            assert flow_result['tokens']['access_token'] == "claude-flow-cross-service-token"
                            assert flow_result['session']['session_id'] == "cross-service-session"
    
    @pytest.mark.asyncio
    async def test_unified_session_management(self):
        """Test unified session management across Claude services."""
        # Mock shared session store
        unified_sessions = {
            "claude-user-123": [
                {
                    'session_id': 'code-desktop-session',
                    'service': 'claude-code',
                    'client_type': 'desktop',
                    'created_at': '2024-01-01T10:00:00Z',
                    'last_activity': '2024-01-01T14:30:00Z'
                },
                {
                    'session_id': 'flow-orchestrator-session',
                    'service': 'claude-flow',
                    'client_type': 'orchestrator',
                    'created_at': '2024-01-01T11:00:00Z',
                    'last_activity': '2024-01-01T14:25:00Z'
                },
                {
                    'session_id': 'code-web-session',
                    'service': 'claude-code',
                    'client_type': 'web',
                    'created_at': '2024-01-01T12:00:00Z',
                    'last_activity': '2024-01-01T14:20:00Z'
                }
            ]
        }
        
        # Test session retrieval across services
        user_sessions = unified_sessions["claude-user-123"]
        
        # Verify cross-service session management
        claude_code_sessions = [s for s in user_sessions if s['service'] == 'claude-code']
        claude_flow_sessions = [s for s in user_sessions if s['service'] == 'claude-flow']
        
        assert len(claude_code_sessions) == 2  # Desktop and web
        assert len(claude_flow_sessions) == 1   # Orchestrator
        assert len(user_sessions) == 3          # Total sessions
        
        # Verify session types
        session_types = [s['client_type'] for s in user_sessions]
        assert 'desktop' in session_types
        assert 'orchestrator' in session_types
        assert 'web' in session_types


if __name__ == "__main__":
    # Run Claude integration tests
import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "code":
            pytest.main(["-v", "TestClaudeCodeIntegration"])
        elif test_category == "flow":
            pytest.main(["-v", "TestClaudeFlowIntegration"])
        elif test_category == "security":
            pytest.main(["-v", "TestClaudeSecurityCompliance"])
        elif test_category == "cross":
            pytest.main(["-v", "TestCrossServiceAuthentication"])
        else:
            print("Available test categories: code, flow, security, cross")
    else:
        # Run all Claude integration tests
        pytest.main(["-v", __file__])
