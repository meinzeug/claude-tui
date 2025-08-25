#!/usr/bin/env python3
"""
Claude-TIU Main Entry Point.

This module provides both CLI and TUI entry points for the application.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from claude_tiu import __version__
from ..core.config_manager import ConfigManager
from ..ui.main_app import ClaudeTIUApp
from ..core.logger import setup_logging
from ..utils.system_check import SystemChecker

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
@click.pass_context
def cli(
    ctx: click.Context,
    version: bool,
    debug: bool,
    config_dir: Optional[Path],
    project_dir: Optional[Path]
) -> None:
    """
    Claude-TIU: Intelligent AI-powered Terminal User Interface.
    
    Launch the TUI application or run CLI commands for advanced software development.
    """
    if version:
        console.print(f"Claude-TIU version {__version__}", style="bold green")
        sys.exit(0)
    
    # Setup logging
    setup_logging(debug=debug)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['config_dir'] = config_dir
    ctx.obj['project_dir'] = project_dir
    
    # If no subcommand, launch TUI
    if ctx.invoked_subcommand is None:
        asyncio.run(launch_tui(debug, config_dir, project_dir))


@cli.command()
@click.argument('template_name')
@click.argument('project_name')
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    help='Output directory for the new project'
)
@click.pass_context
def create(ctx: click.Context, template_name: str, project_name: str, output_dir: Path) -> None:
    """
    Create a new project from a template.
    
    TEMPLATE_NAME: Name of the project template to use
    PROJECT_NAME: Name for the new project
    """
    asyncio.run(_create_project_cli(template_name, project_name, output_dir, ctx.obj))


@cli.command()
@click.argument('prompt', nargs=-1, required=True)
@click.option(
    '--context-files',
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help='Files to include as context'
)
@click.pass_context  
def ask(ctx: click.Context, prompt: tuple[str, ...], context_files: tuple[Path, ...]) -> None:
    """
    Ask Claude a question with optional context files.
    
    PROMPT: The question or request to send to Claude
    """
    prompt_text = ' '.join(prompt)
    asyncio.run(_ask_claude_cli(prompt_text, list(context_files), ctx.obj))


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """
    Run system diagnostics to check for issues.
    """
    asyncio.run(_run_doctor(ctx.obj))


@cli.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--variables',
    help='JSON string of workflow variables'
)
@click.pass_context
def workflow(ctx: click.Context, workflow_file: Path, variables: Optional[str]) -> None:
    """
    Execute a Claude Flow workflow.
    
    WORKFLOW_FILE: Path to the workflow definition file
    """
    asyncio.run(_run_workflow_cli(workflow_file, variables, ctx.obj))


async def launch_tui(
    debug: bool,
    config_dir: Optional[Path],
    project_dir: Optional[Path]
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
        app = ClaudeTIUApp(
            config_manager=config_manager,
            debug=debug,
            initial_project_dir=project_dir
        )
        
        await app.run_async()
        
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="bold blue")
    except Exception as e:
        logger.exception("Fatal error in TUI application")
        console.print(f"❌ Fatal error: {e}", style="red")
        if debug:
            console.print_exception()
        sys.exit(1)


async def _create_project_cli(
    template_name: str,
    project_name: str, 
    output_dir: Path,
    ctx_obj: dict
) -> None:
    """
    CLI implementation for project creation.
    """
    from ..core.project_manager import ProjectManager
    from ..core.config_manager import ConfigManager
    
    try:
        config_manager = ConfigManager(config_dir=ctx_obj.get('config_dir'))
        await config_manager.initialize()
        
        project_manager = ProjectManager(config_manager)
        
        with console.status(f"Creating project '{project_name}' from template '{template_name}'..."):
            project = await project_manager.create_project(
                template_name=template_name,
                project_name=project_name,
                output_directory=output_dir
            )
        
        console.print(f"✅ Project '{project_name}' created successfully!", style="bold green")
        console.print(f"📁 Location: {project.path}", style="blue")
        
    except Exception as e:
        console.print(f"❌ Failed to create project: {e}", style="red")
        if ctx_obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def _ask_claude_cli(
    prompt: str,
    context_files: list[Path],
    ctx_obj: dict
) -> None:
    """
    CLI implementation for asking Claude questions.
    """
    from ..core.ai_interface import AIInterface
    from ..core.config_manager import ConfigManager
    
    try:
        config_manager = ConfigManager(config_dir=ctx_obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Build context from files
        context = {}
        for file_path in context_files:
            context[str(file_path)] = file_path.read_text(encoding='utf-8')
        
        with console.status("Asking Claude..."):
            response = await ai_interface.execute_claude_code(
                prompt=prompt,
                context=context
            )
        
        console.print("🤖 Claude's response:", style="bold blue")
        console.print(response.content)
        
    except Exception as e:
        console.print(f"❌ Failed to get response from Claude: {e}", style="red")
        if ctx_obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def _run_doctor(ctx_obj: dict) -> None:
    """
    CLI implementation for system diagnostics.
    """
    try:
        checker = SystemChecker()
        
        with console.status("Running system diagnostics..."):
            result = await checker.run_comprehensive_check()
        
        # Display results
        if result.all_passed:
            console.print("✅ All system checks passed!", style="bold green")
        else:
            console.print("⚠️ System check results:", style="yellow")
        
        for category, checks in result.categories.items():
            console.print(f"\n📋 {category}:", style="bold")
            for check in checks:
                status = "✅" if check.passed else "❌"
                console.print(f"  {status} {check.name}: {check.message}")
        
        if result.recommendations:
            console.print("\n💡 Recommendations:", style="bold blue")
            for rec in result.recommendations:
                console.print(f"  • {rec}")
        
    except Exception as e:
        console.print(f"❌ Failed to run diagnostics: {e}", style="red")
        if ctx_obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def _run_workflow_cli(
    workflow_file: Path,
    variables: Optional[str],
    ctx_obj: dict
) -> None:
    """
    CLI implementation for workflow execution.
    """
    import json
    from ..core.task_engine import TaskEngine
    from ..core.config_manager import ConfigManager
    
    try:
        config_manager = ConfigManager(config_dir=ctx_obj.get('config_dir'))
        await config_manager.initialize()
        
        task_engine = TaskEngine(config_manager)
        
        # Parse variables if provided
        workflow_vars = {}
        if variables:
            workflow_vars = json.loads(variables)
        
        with console.status(f"Executing workflow from {workflow_file}..."):
            result = await task_engine.execute_workflow_from_file(
                workflow_file=workflow_file,
                variables=workflow_vars
            )
        
        if result.success:
            console.print("✅ Workflow completed successfully!", style="bold green")
        else:
            console.print("❌ Workflow failed:", style="red")
            console.print(result.error_message)
        
        # Show summary
        console.print(f"\n📊 Summary:")
        console.print(f"  • Total tasks: {result.total_tasks}")
        console.print(f"  • Completed: {result.completed_tasks}")
        console.print(f"  • Failed: {result.failed_tasks}")
        console.print(f"  • Duration: {result.duration:.2f}s")
        
    except Exception as e:
        console.print(f"❌ Failed to execute workflow: {e}", style="red")
        if ctx_obj.get('debug'):
            console.print_exception()
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the claude-tiu application.
    """
    cli()


if __name__ == "__main__":
    main()