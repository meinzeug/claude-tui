#!/usr/bin/env python3
"""
Claude-TUI Enhanced CLI Main Entry Point.

Comprehensive command-line interface with AI integration,
project management, and development workflow automation.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from claude_tui import __version__
from ..core.config_manager import ConfigManager
from ..ui.main_app import ClaudeTUIApp
from ..core.logger import setup_logging
from ..utils.system_check import SystemChecker

# Import all command modules
from .commands.core_commands import core_commands
from .commands.ai_commands import ai_commands
from .commands.workspace_commands import workspace_commands
from .commands.integration_commands import integration_commands

console = Console()
logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    help="Show version information"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging"
)
@click.option(
    "--config-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Custom configuration directory"
)
@click.option(
    "--project-dir", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Project directory to open"
)
@click.option(
    "--no-tui",
    is_flag=True,
    help="Disable TUI launch when no command is specified"
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run in headless mode (no interactive UI)"
)
@click.option(
    "--test-mode",
    is_flag=True,
    help="Run in test mode (non-blocking)"
)
@click.pass_context
def cli(
    ctx: click.Context,
    version: bool,
    debug: bool,
    config_dir: Optional[Path],
    project_dir: Optional[Path],
    no_tui: bool,
    headless: bool,
    test_mode: bool
) -> None:
    """
    Claude-TUI: Intelligent AI-powered Terminal User Interface.
    
    Advanced command-line interface for AI-assisted software development,
    project management, and workflow automation.
    
    Examples:
        claude-tui                      # Launch TUI interface
        claude-tui init my-project      # Initialize new project  
        claude-tui ai generate --help   # AI code generation help
        claude-tui workspace list       # List workspaces
        claude-tui build --watch        # Build with file watching
    """
    if version:
        _show_version_info()
        sys.exit(0)
    
    # Setup logging
    setup_logging(debug=debug)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj.update({
        'debug': debug,
        'config_dir': config_dir,
        'project_dir': project_dir,
        'console': console,
        'headless': headless,
        'test_mode': test_mode
    })
    
    # If no subcommand and TUI is enabled, launch TUI
    if ctx.invoked_subcommand is None and not no_tui:
        asyncio.run(launch_tui(debug, config_dir, project_dir, headless, test_mode))


def _show_version_info() -> None:
    """Display comprehensive version information."""
    table = Table(title="Claude-TUI Version Information", show_header=False)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    
    table.add_row("Claude-TUI", __version__)
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    try:
        import textual
        table.add_row("Textual", textual.__version__)
    except ImportError:
        table.add_row("Textual", "Not installed")
    
    try:
        import rich
        table.add_row("Rich", rich.__version__)
    except ImportError:
        table.add_row("Rich", "Not installed")
    
    console.print(table)


async def launch_tui(
    debug: bool,
    config_dir: Optional[Path],
    project_dir: Optional[Path],
    headless: bool = False,
    test_mode: bool = False
) -> None:
    """
    Launch the main TUI application.
    """
    try:
        # System checks
        checker = SystemChecker()
        check_result = await checker.run_checks()
        
        if not check_result.all_passed:
            console.print("⚠️ System check warnings:", style="yellow")
            for warning in check_result.warnings:
                console.print(f"  • {warning}", style="yellow")
            
            if check_result.errors:
                console.print("❌ System check errors:", style="red")
                for error in check_result.errors:
                    console.print(f"  • {error}", style="red")
                console.print("\nPlease fix these errors before proceeding.")
                sys.exit(1)
        
        # Initialize configuration
        config_manager = ConfigManager(config_dir=config_dir)
        await config_manager.initialize()
        
        # Create and run the TUI application
        app = ClaudeTUIApp(
            config_manager=config_manager,
            debug=debug,
            initial_project_dir=project_dir,
            headless=headless,
            test_mode=test_mode
        )
        
        if headless or test_mode:
            await app.init_async()
        else:
            await app.run_async()
        
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="bold blue")
    except Exception as e:
        logger.exception("Fatal error in TUI application")
        console.print(f"❌ Fatal error: {e}", style="red")
        if debug:
            console.print_exception()
        sys.exit(1)


# Import completion commands
from .completion import completion

# Register all command groups
cli.add_command(core_commands, name='core')
cli.add_command(ai_commands, name='ai')
cli.add_command(workspace_commands, name='workspace')
cli.add_command(integration_commands, name='integration')
cli.add_command(completion, name='completion')

# Register convenience aliases at top level
@cli.command()
@click.argument('project_name')
@click.option('--template', default='default', help='Project template to use')
@click.option('--path', type=click.Path(), help='Project directory path')
@click.pass_context
def init(ctx: click.Context, project_name: str, template: str, path: Optional[str]) -> None:
    """Initialize a new project (alias for core init)."""
    from .commands.core_commands import init_project
    asyncio.run(init_project(ctx, project_name, template, path))


@cli.command()
@click.option('--watch', is_flag=True, help='Watch for file changes')
@click.option('--production', is_flag=True, help='Production build')
@click.pass_context  
def build(ctx: click.Context, watch: bool, production: bool) -> None:
    """Build the project (alias for core build)."""
    from .commands.core_commands import build_project
    asyncio.run(build_project(ctx, watch, production))


@cli.command()
@click.option('--coverage', is_flag=True, help='Run with coverage reporting')
@click.option('--watch', is_flag=True, help='Watch for file changes')
@click.pass_context
def test(ctx: click.Context, coverage: bool, watch: bool) -> None:
    """Run tests (alias for core test)."""
    from .commands.core_commands import run_tests
    asyncio.run(run_tests(ctx, coverage, watch))


@cli.command()
@click.option('--environment', default='staging', help='Deployment environment')
@click.option('--dry-run', is_flag=True, help='Show what would be deployed')
@click.pass_context
def deploy(ctx: click.Context, environment: str, dry_run: bool) -> None:
    """Deploy the project (alias for core deploy)."""
    from .commands.core_commands import deploy_project
    asyncio.run(deploy_project(ctx, environment, dry_run))


def main() -> None:
    """
    Main entry point for the claude-tui CLI application.
    """
    cli()


if __name__ == "__main__":
    main()