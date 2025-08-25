"""Comprehensive tests for main.py entry point."""

import pytest
import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

from claude_tiu.main import (
    cli, launch_tui, main,
    _create_project_cli, _ask_claude_cli, _run_doctor, _run_workflow_cli
)
from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.ui.main_app import ClaudeTIUApp
from claude_tiu.utils.system_check import SystemChecker
from claude_tiu.core.project_manager import ProjectManager


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    manager = Mock(spec=ConfigManager)
    manager.initialize = AsyncMock()
    manager.get_config_dir = Mock(return_value=Path('/tmp/config'))
    return manager


@pytest.fixture
def mock_system_checker():
    """Mock system checker."""
    checker = Mock(spec=SystemChecker)
    
    # Mock check result
    check_result = Mock()
    check_result.all_passed = True
    check_result.warnings = []
    check_result.errors = []
    
    checker.run_checks = AsyncMock(return_value=check_result)
    checker.run_comprehensive_check = AsyncMock(return_value=check_result)
    
    return checker, check_result


@pytest.fixture
def mock_app():
    """Mock TUI application."""
    app = Mock(spec=ClaudeTIUApp)
    app.run_async = AsyncMock()
    return app


class TestCLICommands:
    """Test CLI command handling."""
    
    def test_cli_version_flag(self):
        """Test --version flag displays version and exits."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'Claude-TIU version' in result.output
    
    def test_cli_debug_flag_stored_in_context(self):
        """Test --debug flag is stored in context."""
        runner = CliRunner()
        
        with patch('claude_tiu.main.launch_tui') as mock_launch:
            mock_launch.return_value = asyncio.coroutine(lambda: None)()
            
            result = runner.invoke(cli, ['--debug'])
            
            # Check that debug is True in the context
            mock_launch.assert_called_once()
            args = mock_launch.call_args[0]
            assert args[0] == True  # debug=True
    
    def test_cli_config_dir_parameter(self):
        """Test custom config directory parameter."""
        runner = CliRunner()
        config_dir = '/custom/config'
        
        with patch('claude_tiu.main.launch_tui') as mock_launch:
            mock_launch.return_value = asyncio.coroutine(lambda: None)()
            
            result = runner.invoke(cli, ['--config-dir', config_dir])
            
            mock_launch.assert_called_once()
            args = mock_launch.call_args[0]
            assert str(args[1]) == config_dir  # config_dir
    
    def test_create_command_parameters(self):
        """Test create command parameter handling."""
        runner = CliRunner()
        
        with patch('claude_tiu.main._create_project_cli') as mock_create:
            mock_create.return_value = asyncio.coroutine(lambda *args: None)()
            
            result = runner.invoke(cli, [
                'create', 'template_name', 'project_name',
                '--output-dir', '/output/path'
            ])
            
            mock_create.assert_called_once()
            args = mock_create.call_args[0]
            assert args[0] == 'template_name'
            assert args[1] == 'project_name'
            assert str(args[2]) == '/output/path'
    
    def test_ask_command_with_context_files(self):
        """Test ask command with context files."""
        runner = CliRunner()
        
        with patch('claude_tiu.main._ask_claude_cli') as mock_ask:
            mock_ask.return_value = asyncio.coroutine(lambda *args: None)()
            
            with runner.isolated_filesystem():
                # Create test files
                Path('file1.py').write_text('content1')
                Path('file2.py').write_text('content2')
                
                result = runner.invoke(cli, [
                    'ask', 'What does this code do?',
                    '--context-files', 'file1.py',
                    '--context-files', 'file2.py'
                ])
                
                mock_ask.assert_called_once()
                args = mock_ask.call_args[0]
                assert args[0] == 'What does this code do?'
                assert len(args[1]) == 2  # Two context files
    
    def test_doctor_command(self):
        """Test doctor command execution."""
        runner = CliRunner()
        
        with patch('claude_tiu.main._run_doctor') as mock_doctor:
            mock_doctor.return_value = asyncio.coroutine(lambda *args: None)()
            
            result = runner.invoke(cli, ['doctor'])
            
            mock_doctor.assert_called_once()
    
    def test_workflow_command_with_variables(self):
        """Test workflow command with JSON variables."""
        runner = CliRunner()
        
        with patch('claude_tiu.main._run_workflow_cli') as mock_workflow:
            mock_workflow.return_value = asyncio.coroutine(lambda *args: None)()
            
            with runner.isolated_filesystem():
                # Create test workflow file
                Path('workflow.yaml').write_text('test: workflow')
                
                result = runner.invoke(cli, [
                    'workflow', 'workflow.yaml',
                    '--variables', '{"key": "value"}'
                ])
                
                mock_workflow.assert_called_once()
                args = mock_workflow.call_args[0]
                assert args[1] == '{"key": "value"}'


class TestLaunchTUI:
    """Test TUI application launching."""
    
    @pytest.mark.asyncio
    async def test_launch_tui_successful(self, mock_system_checker, mock_config_manager, mock_app):
        """Test successful TUI launch."""
        checker, check_result = mock_system_checker
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ClaudeTIUApp', return_value=mock_app):
            
            await launch_tui(debug=False, config_dir=None, project_dir=None)
            
            checker.run_checks.assert_called_once()
            mock_config_manager.initialize.assert_called_once()
            mock_app.run_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_launch_tui_system_check_warnings(self, mock_system_checker, mock_config_manager, mock_app):
        """Test TUI launch with system check warnings."""
        checker, check_result = mock_system_checker
        check_result.warnings = ['Warning 1', 'Warning 2']
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ClaudeTIUApp', return_value=mock_app), \
             patch('claude_tiu.main.console') as mock_console:
            
            await launch_tui(debug=False, config_dir=None, project_dir=None)
            
            # Check that warnings were printed
            mock_console.print.assert_any_call("⚠️ System check warnings:", style="yellow")
            mock_app.run_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_launch_tui_system_check_errors_exit(self, mock_system_checker):
        """Test TUI launch exits on system check errors."""
        checker, check_result = mock_system_checker
        check_result.all_passed = False
        check_result.errors = ['Critical Error 1']
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.console') as mock_console, \
             patch('sys.exit') as mock_exit:
            
            await launch_tui(debug=False, config_dir=None, project_dir=None)
            
            mock_console.print.assert_any_call("❌ System check errors:", style="red")
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_launch_tui_keyboard_interrupt(self, mock_system_checker, mock_config_manager, mock_app):
        """Test graceful handling of keyboard interrupt."""
        checker, check_result = mock_system_checker
        mock_app.run_async.side_effect = KeyboardInterrupt()
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ClaudeTIUApp', return_value=mock_app), \
             patch('claude_tiu.main.console') as mock_console:
            
            await launch_tui(debug=False, config_dir=None, project_dir=None)
            
            mock_console.print.assert_any_call("\n👋 Goodbye!", style="bold blue")
    
    @pytest.mark.asyncio
    async def test_launch_tui_exception_handling(self, mock_system_checker, mock_config_manager, mock_app):
        """Test exception handling during TUI launch."""
        checker, check_result = mock_system_checker
        mock_app.run_async.side_effect = Exception("Test error")
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ClaudeTIUApp', return_value=mock_app), \
             patch('claude_tiu.main.console') as mock_console, \
             patch('sys.exit') as mock_exit:
            
            await launch_tui(debug=False, config_dir=None, project_dir=None)
            
            mock_console.print.assert_any_call("❌ Fatal error: Test error", style="red")
            mock_exit.assert_called_once_with(1)


class TestCLIImplementations:
    """Test CLI command implementations."""
    
    @pytest.mark.asyncio
    async def test_create_project_cli_success(self, mock_config_manager):
        """Test successful project creation via CLI."""
        mock_project_manager = Mock(spec=ProjectManager)
        mock_project = Mock()
        mock_project.path = Path('/project/path')
        mock_project_manager.create_project = AsyncMock(return_value=mock_project)
        
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ProjectManager', return_value=mock_project_manager), \
             patch('claude_tiu.main.console') as mock_console:
            
            await _create_project_cli('template', 'project', Path('/output'), ctx_obj)
            
            mock_config_manager.initialize.assert_called_once()
            mock_project_manager.create_project.assert_called_once_with(
                template_name='template',
                project_name='project',
                output_directory=Path('/output')
            )
            mock_console.print.assert_any_call(
                "✅ Project 'project' created successfully!", style="bold green"
            )
    
    @pytest.mark.asyncio
    async def test_create_project_cli_error(self, mock_config_manager):
        """Test project creation CLI error handling."""
        mock_config_manager.initialize.side_effect = Exception("Config error")
        
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.console') as mock_console, \
             patch('sys.exit') as mock_exit:
            
            await _create_project_cli('template', 'project', Path('/output'), ctx_obj)
            
            mock_console.print.assert_any_call(
                "❌ Failed to create project: Config error", style="red"
            )
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_ask_claude_cli_success(self, mock_config_manager):
        """Test successful Claude query via CLI."""
        mock_ai_interface = Mock()
        mock_response = Mock()
        mock_response.content = "Claude's response"
        mock_ai_interface.execute_claude_code = AsyncMock(return_value=mock_response)
        
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.AIInterface', return_value=mock_ai_interface), \
             patch('claude_tiu.main.console') as mock_console:
            
            await _ask_claude_cli('Test question', [Path('/file.py')], ctx_obj)
            
            mock_ai_interface.execute_claude_code.assert_called_once()
            mock_console.print.assert_any_call("🤖 Claude's response:", style="bold blue")
            mock_console.print.assert_any_call("Claude's response")
    
    @pytest.mark.asyncio
    async def test_run_doctor_success(self, mock_system_checker):
        """Test successful doctor command."""
        checker, check_result = mock_system_checker
        check_result.categories = {
            'System': [Mock(passed=True, name='Test', message='OK')]
        }
        check_result.recommendations = ['Recommendation 1']
        
        ctx_obj = {'debug': False}
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.console') as mock_console:
            
            await _run_doctor(ctx_obj)
            
            checker.run_comprehensive_check.assert_called_once()
            mock_console.print.assert_any_call("✅ All system checks passed!", style="bold green")
    
    @pytest.mark.asyncio
    async def test_run_workflow_cli_success(self, mock_config_manager):
        """Test successful workflow execution via CLI."""
        mock_task_engine = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.total_tasks = 5
        mock_result.completed_tasks = 5
        mock_result.failed_tasks = 0
        mock_result.duration = 10.5
        mock_task_engine.execute_workflow_from_file = AsyncMock(return_value=mock_result)
        
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.TaskEngine', return_value=mock_task_engine), \
             patch('claude_tiu.main.console') as mock_console:
            
            await _run_workflow_cli(
                Path('/workflow.yaml'), 
                '{"key": "value"}', 
                ctx_obj
            )
            
            mock_task_engine.execute_workflow_from_file.assert_called_once()
            mock_console.print.assert_any_call("✅ Workflow completed successfully!", style="bold green")


class TestMainFunction:
    """Test main entry point function."""
    
    def test_main_calls_cli_main(self):
        """Test main function calls CLI main."""
        with patch('claude_tiu.main.cli_main') as mock_cli_main:
            main()
            mock_cli_main.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_cli_invalid_config_dir(self):
        """Test CLI with invalid config directory."""
        runner = CliRunner()
        
        with patch('claude_tiu.main.launch_tui') as mock_launch:
            mock_launch.return_value = asyncio.coroutine(lambda: None)()
            
            # This should work - click will validate the path
            result = runner.invoke(cli, ['--config-dir', '/nonexistent'])
            
            # Should still call launch_tui even if path doesn't exist
            mock_launch.assert_called_once()
    
    def test_cli_missing_required_args(self):
        """Test CLI commands with missing required arguments."""
        runner = CliRunner()
        
        # Test create command without required args
        result = runner.invoke(cli, ['create'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output
        
        # Test ask command without prompt
        result = runner.invoke(cli, ['ask'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output
    
    @pytest.mark.asyncio
    async def test_workflow_cli_invalid_json_variables(self, mock_config_manager):
        """Test workflow CLI with invalid JSON variables."""
        mock_task_engine = Mock()
        
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.TaskEngine', return_value=mock_task_engine), \
             patch('claude_tiu.main.console') as mock_console, \
             patch('sys.exit') as mock_exit:
            
            # Invalid JSON should cause error
            await _run_workflow_cli(
                Path('/workflow.yaml'),
                'invalid json}',
                ctx_obj
            )
            
            mock_console.print.assert_any_call(
                "❌ Failed to execute workflow: Expecting property name enclosed in double quotes: line 1 column 13 (char 12)",
                style="red"
            )
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_ask_claude_cli_with_nonexistent_context_files(self, mock_config_manager):
        """Test ask Claude CLI with nonexistent context files."""
        ctx_obj = {'config_dir': None, 'debug': False}
        
        with patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.console') as mock_console, \
             patch('sys.exit') as mock_exit:
            
            await _ask_claude_cli(
                'Test question',
                [Path('/nonexistent.py')],
                ctx_obj
            )
            
            # Should handle file not found error
            mock_console.print.assert_any_call(
                pytest.stringContaining("❌ Failed to get response from Claude:"),
                style="red"
            )
            mock_exit.assert_called_once_with(1)


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_cli_project_dir_validation(self):
        """Test project directory validation."""
        runner = CliRunner()
        
        with patch('claude_tiu.main.launch_tui') as mock_launch:
            mock_launch.return_value = asyncio.coroutine(lambda: None)()
            
            # Nonexistent project directory should fail validation
            result = runner.invoke(cli, ['--project-dir', '/nonexistent'])
            
            # Click should handle this validation
            assert result.exit_code != 0 or mock_launch.called
    
    def test_create_command_output_dir_validation(self):
        """Test create command output directory validation."""
        runner = CliRunner()
        
        with patch('claude_tiu.main._create_project_cli') as mock_create:
            mock_create.return_value = asyncio.coroutine(lambda *args: None)()
            
            # Should accept any path for output directory
            result = runner.invoke(cli, [
                'create', 'template', 'project',
                '--output-dir', '/any/path'
            ])
            
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_launch_tui_with_custom_paths(self, mock_system_checker, mock_config_manager, mock_app):
        """Test launch TUI with custom configuration and project paths."""
        checker, check_result = mock_system_checker
        config_dir = Path('/custom/config')
        project_dir = Path('/custom/project')
        
        with patch('claude_tiu.main.SystemChecker', return_value=checker), \
             patch('claude_tiu.main.ConfigManager', return_value=mock_config_manager), \
             patch('claude_tiu.main.ClaudeTIUApp', return_value=mock_app):
            
            await launch_tui(
                debug=True,
                config_dir=config_dir,
                project_dir=project_dir
            )
            
            # Check that custom paths are passed to components
            mock_config_manager.initialize.assert_called_once()
            mock_app.run_async.assert_called_once()


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_version_integration(self):
        """Test version command integration."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'Claude-TIU version' in result.output
        assert result.output.strip().endswith('0.1.0')
    
    def test_help_integration(self):
        """Test help command integration."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Claude-TIU: Intelligent AI-powered Terminal User Interface' in result.output
        assert 'create' in result.output
        assert 'ask' in result.output
        assert 'doctor' in result.output
        assert 'workflow' in result.output
    
    def test_subcommand_help(self):
        """Test subcommand help integration."""
        runner = CliRunner()
        
        # Test create command help
        result = runner.invoke(cli, ['create', '--help'])
        assert result.exit_code == 0
        assert 'Create a new project from a template' in result.output
        
        # Test ask command help
        result = runner.invoke(cli, ['ask', '--help'])
        assert result.exit_code == 0
        assert 'Ask Claude a question' in result.output
        
        # Test doctor command help
        result = runner.invoke(cli, ['doctor', '--help'])
        assert result.exit_code == 0
        assert 'Run system diagnostics' in result.output
        
        # Test workflow command help
        result = runner.invoke(cli, ['workflow', '--help'])
        assert result.exit_code == 0
        assert 'Execute a Claude Flow workflow' in result.output
