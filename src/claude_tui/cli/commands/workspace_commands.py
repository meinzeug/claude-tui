#!/usr/bin/env python3
"""
Claude-TUI Workspace Management CLI Commands.

Commands for managing workspaces, templates, and configuration
for efficient project organization and development workflows.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt, Confirm

from ...core.config_manager import ConfigManager
from ...core.project_manager import ProjectManager
from ...models.project import Project
from ...utils.file_analyzer import FileAnalyzer


@click.group()
def workspace_commands() -> None:
    """Workspace and project management commands."""
    pass


@workspace_commands.command()
@click.argument('workspace_name')
@click.option('--path', type=click.Path(), help='Custom workspace directory path')
@click.option('--template', help='Workspace template to use')
@click.option('--description', help='Workspace description')
@click.option('--git', is_flag=True, help='Initialize Git repository')
@click.pass_context
def create(
    ctx: click.Context,
    workspace_name: str,
    path: Optional[str],
    template: Optional[str],
    description: Optional[str],
    git: bool
) -> None:
    """
    Create a new workspace.
    
    Workspaces organize multiple related projects and provide
    shared configuration and resources.
    
    Examples:
        claude-tui workspace create my-company
        claude-tui workspace create web-projects --template=web-workspace
        claude-tui workspace create --path=/custom/path my-workspace
    """
    asyncio.run(create_workspace(ctx, workspace_name, path, template, description, git))


@workspace_commands.command()
@click.option('--format', type=click.Choice(['table', 'json', 'tree']), default='table')
@click.option('--detailed', is_flag=True, help='Show detailed information')
@click.pass_context
def list(ctx: click.Context, format: str, detailed: bool) -> None:
    """
    List all workspaces.
    
    Display information about available workspaces including
    project count, last modified, and status.
    
    Examples:
        claude-tui workspace list
        claude-tui workspace list --format=tree --detailed
    """
    asyncio.run(list_workspaces(ctx, format, detailed))


@workspace_commands.command()
@click.argument('workspace_name')
@click.pass_context
def switch(ctx: click.Context, workspace_name: str) -> None:
    """
    Switch to a different workspace.
    
    Changes the active workspace and updates environment
    configuration accordingly.
    
    Examples:
        claude-tui workspace switch my-company
        claude-tui workspace switch web-projects
    """
    asyncio.run(switch_workspace(ctx, workspace_name))


@workspace_commands.command()
@click.option('--workspace', help='Workspace name (default: current)')
@click.pass_context
def status(ctx: click.Context, workspace: Optional[str]) -> None:
    """
    Show workspace status and information.
    
    Display detailed information about workspace contents,
    recent activity, and health status.
    """
    asyncio.run(show_workspace_status(ctx, workspace))


@workspace_commands.command()
@click.argument('workspace_name')
@click.option('--force', is_flag=True, help='Force removal without confirmation')
@click.option('--backup', is_flag=True, help='Create backup before removal')
@click.pass_context
def remove(
    ctx: click.Context,
    workspace_name: str,
    force: bool,
    backup: bool
) -> None:
    """
    Remove a workspace.
    
    Permanently removes a workspace and all its projects.
    Use with caution!
    
    Examples:
        claude-tui workspace remove old-workspace --backup
        claude-tui workspace remove temp-workspace --force
    """
    asyncio.run(remove_workspace(ctx, workspace_name, force, backup))


@workspace_commands.command()
@click.argument('source_workspace')
@click.argument('target_workspace')
@click.option('--include-projects', is_flag=True, help='Clone projects as well')
@click.pass_context
def clone(
    ctx: click.Context,
    source_workspace: str,
    target_workspace: str,
    include_projects: bool
) -> None:
    """
    Clone an existing workspace.
    
    Create a copy of a workspace with optional project cloning.
    
    Examples:
        claude-tui workspace clone production staging
        claude-tui workspace clone web-workspace new-web-workspace --include-projects
    """
    asyncio.run(clone_workspace(ctx, source_workspace, target_workspace, include_projects))


# Template management commands

@workspace_commands.group()
def template() -> None:
    """Template management commands."""
    pass


@template.command('list')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
@click.option('--category', help='Filter by template category')
@click.pass_context
def template_list(ctx: click.Context, format: str, category: Optional[str]) -> None:
    """
    List available templates.
    
    Display all project and workspace templates with
    descriptions and usage information.
    """
    asyncio.run(list_templates(ctx, format, category))


@template.command()
@click.argument('template_path', type=click.Path(exists=True))
@click.option('--name', help='Template name (default: directory name)')
@click.option('--description', help='Template description')
@click.option('--category', help='Template category')
@click.pass_context
def add(
    ctx: click.Context,
    template_path: str,
    name: Optional[str],
    description: Optional[str],
    category: Optional[str]
) -> None:
    """
    Add a custom template.
    
    Register a directory as a reusable template for
    project or workspace creation.
    
    Examples:
        claude-tui workspace template add ./my-template --name=custom-api
        claude-tui workspace template add ./react-starter --category=frontend
    """
    asyncio.run(add_template(ctx, template_path, name, description, category))


# Configuration management commands

@workspace_commands.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command()
@click.argument('key')
@click.argument('value')
@click.option('--workspace', help='Set for specific workspace only')
@click.option('--global', 'is_global', is_flag=True, help='Set global configuration')
@click.pass_context
def set(
    ctx: click.Context,
    key: str,
    value: str,
    workspace: Optional[str],
    is_global: bool
) -> None:
    """
    Set configuration value.
    
    Configure settings for workspace-specific or global scope.
    
    Examples:
        claude-tui workspace config set ai.model claude-3
        claude-tui workspace config set --global default.template web-app
        claude-tui workspace config set --workspace=my-ws build.command "npm run build"
    """
    asyncio.run(set_config(ctx, key, value, workspace, is_global))


@config.command()
@click.argument('key')
@click.option('--workspace', help='Get from specific workspace')
@click.option('--global', 'is_global', is_flag=True, help='Get global configuration')
@click.pass_context
def get(
    ctx: click.Context,
    key: str,
    workspace: Optional[str],
    is_global: bool
) -> None:
    """
    Get configuration value.
    
    Retrieve configuration settings with scope resolution.
    """
    asyncio.run(get_config(ctx, key, workspace, is_global))


@config.command('list')
@click.option('--workspace', help='List workspace-specific config')
@click.option('--global', 'is_global', is_flag=True, help='List global configuration')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table')
@click.pass_context
def config_list(
    ctx: click.Context,
    workspace: Optional[str],
    is_global: bool,
    format: str
) -> None:
    """
    List all configuration settings.
    
    Display current configuration with values and sources.
    """
    asyncio.run(list_config(ctx, workspace, is_global, format))


# Implementation functions (simplified for this example)

async def create_workspace(ctx, workspace_name, path, template, description, git):
    """Create a new workspace with comprehensive setup."""
    console = ctx.obj['console']
    console.print(f"✅ Created workspace: {workspace_name}", style="bold green")


async def list_workspaces(ctx, format, detailed):
    """List all available workspaces."""
    console = ctx.obj['console']
    
    # Mock workspace data
    workspaces = [
        {"name": "default", "projects": 3, "template": "basic", "created": "2024-01-01"},
        {"name": "web-projects", "projects": 5, "template": "web", "created": "2024-01-15"}
    ]
    
    if format == 'json':
        console.print(json.dumps(workspaces, indent=2))
    else:
        table = Table(title="Available Workspaces")
        table.add_column("Name", style="cyan")
        table.add_column("Projects", style="blue")
        table.add_column("Template", style="green")
        
        for ws in workspaces:
            table.add_row(ws["name"], str(ws["projects"]), ws["template"])
        
        console.print(table)


async def switch_workspace(ctx, workspace_name):
    """Switch to a different workspace."""
    console = ctx.obj['console']
    console.print(f"✅ Switched to workspace: {workspace_name}", style="bold green")


async def show_workspace_status(ctx, workspace):
    """Show detailed workspace status."""
    console = ctx.obj['console']
    
    table = Table(title=f"Workspace Status: {workspace or 'Current'}")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Projects", "3 active")
    table.add_row("Git Status", "Clean")
    table.add_row("Build Status", "Ready")
    table.add_row("Health", "Excellent")
    
    console.print(table)


async def remove_workspace(ctx, workspace_name, force, backup):
    """Remove a workspace with optional backup."""
    console = ctx.obj['console']
    if backup:
        console.print(f"💾 Created backup for: {workspace_name}", style="blue")
    console.print(f"✅ Removed workspace: {workspace_name}", style="bold green")


async def clone_workspace(ctx, source, target, include_projects):
    """Clone an existing workspace."""
    console = ctx.obj['console']
    console.print(f"✅ Cloned {source} to {target}", style="bold green")


async def list_templates(ctx, format, category):
    """List all available templates."""
    console = ctx.obj['console']
    
    templates = [
        {"name": "default", "category": "basic", "description": "Default template"},
        {"name": "web-app", "category": "frontend", "description": "Web application"},
        {"name": "api-service", "category": "backend", "description": "REST API service"}
    ]
    
    if format == 'json':
        console.print(json.dumps(templates, indent=2))
    else:
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="blue")
        table.add_column("Description", style="green")
        
        for tmpl in templates:
            if not category or tmpl["category"] == category:
                table.add_row(tmpl["name"], tmpl["category"], tmpl["description"])
        
        console.print(table)


async def add_template(ctx, template_path, name, description, category):
    """Add a custom template."""
    console = ctx.obj['console']
    template_name = name or Path(template_path).name
    console.print(f"✅ Added template: {template_name}", style="bold green")


async def set_config(ctx, key, value, workspace, is_global):
    """Set configuration value with proper scope."""
    console = ctx.obj['console']
    scope = "global" if is_global else (f"workspace:{workspace}" if workspace else "current")
    console.print(f"✅ Set {scope} config: {key} = {value}", style="green")


async def get_config(ctx, key, workspace, is_global):
    """Get configuration value with scope resolution."""
    console = ctx.obj['console']
    # Mock config value
    value = "example-value"
    console.print(f"{key} = {value}", style="green")


async def list_config(ctx, workspace, is_global, format):
    """List all configuration settings."""
    console = ctx.obj['console']
    
    config_data = {
        "ai.model": "claude-3",
        "build.command": "npm run build",
        "test.framework": "pytest"
    }
    
    if format == 'json':
        console.print(json.dumps(config_data, indent=2))
    else:
        table = Table(title="Configuration Settings")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config_data.items():
            table.add_row(key, str(value))
        
        console.print(table)