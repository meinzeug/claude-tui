#!/usr/bin/env python3
"""
Claude-TUI CLI Bash Completion System.

Provides intelligent bash completion for all CLI commands,
options, and dynamic content like workspace names, templates, etc.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from rich.console import Console

console = Console()

# Bash completion script template
BASH_COMPLETION_SCRIPT = '''
_claude_tui_completion() {
    local cur prev words cword
    _init_completion || return

    case "${prev}" in
        --workspace)
            COMPREPLY=($(compgen -W "$(claude-tui workspace list --format=json 2>/dev/null | jq -r '.[].name' 2>/dev/null || echo '')" -- "${cur}"))
            return
            ;;
        --template)
            COMPREPLY=($(compgen -W "$(claude-tui workspace template list --format=json 2>/dev/null | jq -r '.[].name' 2>/dev/null || echo 'default web-app api-service cli-tool library')" -- "${cur}"))
            return
            ;;
        --language)
            COMPREPLY=($(compgen -W "python javascript typescript java go rust cpp c" -- "${cur}"))
            return
            ;;
        --framework)
            COMPREPLY=($(compgen -W "react vue angular express fastapi django flask spring laravel" -- "${cur}"))
            return
            ;;
        --environment)
            COMPREPLY=($(compgen -W "development staging production test" -- "${cur}"))
            return
            ;;
        --format)
            COMPREPLY=($(compgen -W "table json yaml tree chart" -- "${cur}"))
            return
            ;;
        --mode)
            COMPREPLY=($(compgen -W "explain debug suggest analyze" -- "${cur}"))
            return
            ;;
        --provider)
            COMPREPLY=($(compgen -W "github gitlab jenkins travis circleci" -- "${cur}"))
            return
            ;;
        --test-type)
            COMPREPLY=($(compgen -W "unit integration e2e performance" -- "${cur}"))
            return
            ;;
        --target)
            COMPREPLY=($(compgen -W "speed memory size latency throughput" -- "${cur}"))
            return
            ;;
    esac

    # Complete subcommands
    case "${words[1]}" in
        core)
            case "${words[2]}" in
                ""|*)
                    COMPREPLY=($(compgen -W "init build test deploy validate doctor status" -- "${cur}"))
                    ;;
            esac
            ;;
        ai)
            case "${words[2]}" in
                ""|*)
                    COMPREPLY=($(compgen -W "generate review fix optimize test-generate document translate ask" -- "${cur}"))
                    ;;
            esac
            ;;
        workspace)
            case "${words[2]}" in
                template)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "list add remove info" -- "${cur}"))
                            ;;
                    esac
                    ;;
                config)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "set get list import export" -- "${cur}"))
                            ;;
                    esac
                    ;;
                ""|*)
                    COMPREPLY=($(compgen -W "create list switch status remove clone template config" -- "${cur}"))
                    ;;
            esac
            ;;
        integration)
            case "${words[2]}" in
                github)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "setup status create-pr pr create-issue issues workflows" -- "${cur}"))
                            ;;
                    esac
                    ;;
                progress)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "monitor report alert" -- "${cur}"))
                            ;;
                    esac
                    ;;
                batch)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "run execute status" -- "${cur}"))
                            ;;
                    esac
                    ;;
                cicd)
                    case "${words[3]}" in
                        ""|*)
                            COMPREPLY=($(compgen -W "setup trigger status" -- "${cur}"))
                            ;;
                    esac
                    ;;
                ""|*)
                    COMPREPLY=($(compgen -W "github progress batch cicd connect services sync" -- "${cur}"))
                    ;;
            esac
            ;;
        ""|*)
            # Top-level commands
            COMPREPLY=($(compgen -W "core ai workspace integration init build test deploy --version --debug --help" -- "${cur}"))
            ;;
    esac

    # File completion for specific options
    case "${prev}" in
        --config|--output|--script-file|--workflow-file|--context-files)
            COMPREPLY=($(compgen -f -- "${cur}"))
            return
            ;;
        --path|--output-dir|--project-dir|--config-dir)
            COMPREPLY=($(compgen -d -- "${cur}"))
            return
            ;;
    esac

    # Default completion
    if [[ "${cur}" == -* ]]; then
        COMPREPLY=($(compgen -W "--help --debug --version --config-dir --project-dir --no-tui" -- "${cur}"))
    fi
}

complete -F _claude_tui_completion claude-tui
complete -F _claude_tui_completion ctiu
'''

# Fish completion script
FISH_COMPLETION_SCRIPT = '''
# Claude-TUI fish completions

# Core commands
complete -c claude-tui -f
complete -c claude-tui -n '__fish_use_subcommand' -a 'core' -d 'Core project management commands'
complete -c claude-tui -n '__fish_use_subcommand' -a 'ai' -d 'AI-powered development assistance'
complete -c claude-tui -n '__fish_use_subcommand' -a 'workspace' -d 'Workspace and project management'
complete -c claude-tui -n '__fish_use_subcommand' -a 'integration' -d 'External integrations'
complete -c claude-tui -n '__fish_use_subcommand' -a 'init' -d 'Initialize new project (alias)'
complete -c claude-tui -n '__fish_use_subcommand' -a 'build' -d 'Build project (alias)'
complete -c claude-tui -n '__fish_use_subcommand' -a 'test' -d 'Run tests (alias)'
complete -c claude-tui -n '__fish_use_subcommand' -a 'deploy' -d 'Deploy project (alias)'

# Global options
complete -c claude-tui -l version -d 'Show version information'
complete -c claude-tui -l debug -d 'Enable debug logging'
complete -c claude-tui -l config-dir -d 'Custom configuration directory' -r
complete -c claude-tui -l project-dir -d 'Project directory to open' -r
complete -c claude-tui -l no-tui -d 'Disable TUI launch'
complete -c claude-tui -l help -d 'Show help information'

# Core subcommands
complete -c claude-tui -n '__fish_seen_subcommand_from core' -a 'init build test deploy validate doctor status'

# AI subcommands
complete -c claude-tui -n '__fish_seen_subcommand_from ai' -a 'generate review fix optimize test-generate document translate ask'

# Workspace subcommands
complete -c claude-tui -n '__fish_seen_subcommand_from workspace' -a 'create list switch status remove clone template config'

# Integration subcommands
complete -c claude-tui -n '__fish_seen_subcommand_from integration' -a 'github progress batch cicd connect services sync'

# Template options
complete -c claude-tui -l template -d 'Project template' -xa 'default web-app api-service cli-tool library'

# Language options
complete -c claude-tui -l language -d 'Programming language' -xa 'python javascript typescript java go rust cpp c'

# Framework options
complete -c claude-tui -l framework -d 'Framework to use' -xa 'react vue angular express fastapi django flask spring laravel'

# Environment options
complete -c claude-tui -l environment -d 'Deployment environment' -xa 'development staging production test'

# Format options
complete -c claude-tui -l format -d 'Output format' -xa 'table json yaml tree chart'
'''

# Zsh completion script
ZSH_COMPLETION_SCRIPT = '''
#compdef claude-tui

_claude_tui() {
    local -a commands
    local context state line
    
    commands=(
        'core:Core project management commands'
        'ai:AI-powered development assistance'
        'workspace:Workspace and project management'
        'integration:External integrations'
        'init:Initialize new project (alias)'
        'build:Build project (alias)'
        'test:Run tests (alias)'
        'deploy:Deploy project (alias)'
    )
    
    _arguments -C \\
        '--version[Show version information]' \\
        '--debug[Enable debug logging]' \\
        '--config-dir[Custom configuration directory]:directory:_directories' \\
        '--project-dir[Project directory to open]:directory:_directories' \\
        '--no-tui[Disable TUI launch]' \\
        '--help[Show help information]' \\
        '1: :->command' \\
        '*: :->args'
    
    case $state in
        command)
            _describe -t commands 'claude-tui commands' commands
            ;;
        args)
            case $line[1] in
                core)
                    _arguments \\
                        '1: :(init build test deploy validate doctor status)'
                    ;;
                ai)
                    _arguments \\
                        '1: :(generate review fix optimize test-generate document translate ask)'
                    ;;
                workspace)
                    _arguments \\
                        '1: :(create list switch status remove clone template config)'
                    ;;
                integration)
                    _arguments \\
                        '1: :(github progress batch cicd connect services sync)'
                    ;;
            esac
            ;;
    esac
}

_claude_tui "$@"
'''


class CompletionManager:
    """Manages CLI completion system installation and configuration."""
    
    def __init__(self):
        self.home = Path.home()
        self.config_dir = self.home / ".claude-tui"
        self.completion_dir = self.config_dir / "completion"
    
    def install_bash_completion(self) -> bool:
        """Install bash completion script."""
        try:
            self.completion_dir.mkdir(parents=True, exist_ok=True)
            
            # Write bash completion script
            bash_completion_file = self.completion_dir / "claude-tui-completion.bash"
            bash_completion_file.write_text(BASH_COMPLETION_SCRIPT)
            
            # Try to add to bashrc
            bashrc = self.home / ".bashrc"
            source_line = f"source {bash_completion_file}"
            
            if bashrc.exists():
                bashrc_content = bashrc.read_text()
                if source_line not in bashrc_content:
                    with bashrc.open('a') as f:
                        f.write(f"\\n# Claude-TUI completion\\n{source_line}\\n")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to install bash completion: {e}", style="red")
            return False
    
    def install_fish_completion(self) -> bool:
        """Install fish completion script."""
        try:
            # Fish completions directory
            fish_completions_dir = self.home / ".config" / "fish" / "completions"
            fish_completions_dir.mkdir(parents=True, exist_ok=True)
            
            # Write fish completion script
            fish_completion_file = fish_completions_dir / "claude-tui.fish"
            fish_completion_file.write_text(FISH_COMPLETION_SCRIPT)
            
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to install fish completion: {e}", style="red")
            return False
    
    def install_zsh_completion(self) -> bool:
        """Install zsh completion script."""
        try:
            # Zsh completions directory
            zsh_completions_dir = self.home / ".zsh" / "completions"
            zsh_completions_dir.mkdir(parents=True, exist_ok=True)
            
            # Write zsh completion script
            zsh_completion_file = zsh_completions_dir / "_claude-tui"
            zsh_completion_file.write_text(ZSH_COMPLETION_SCRIPT)
            
            # Try to add to zshrc
            zshrc = self.home / ".zshrc"
            fpath_line = f"fpath=({zsh_completions_dir} $fpath)"
            
            if zshrc.exists():
                zshrc_content = zshrc.read_text()
                if str(zsh_completions_dir) not in zshrc_content:
                    with zshrc.open('a') as f:
                        f.write(f"\\n# Claude-TUI completion\\n{fpath_line}\\nautoload -U compinit && compinit\\n")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to install zsh completion: {e}", style="red")
            return False
    
    def detect_shell(self) -> str:
        """Detect the current shell."""
        shell = os.environ.get('SHELL', '')
        if 'bash' in shell:
            return 'bash'
        elif 'zsh' in shell:
            return 'zsh'
        elif 'fish' in shell:
            return 'fish'
        else:
            return 'unknown'
    
    def install_completion(self, shell: Optional[str] = None) -> bool:
        """Install completion for specified or detected shell."""
        if not shell:
            shell = self.detect_shell()
        
        success = False
        
        if shell == 'bash':
            success = self.install_bash_completion()
        elif shell == 'fish':
            success = self.install_fish_completion()
        elif shell == 'zsh':
            success = self.install_zsh_completion()
        elif shell == 'all':
            success = (
                self.install_bash_completion() and
                self.install_fish_completion() and
                self.install_zsh_completion()
            )
        else:
            console.print(f"❌ Unsupported shell: {shell}", style="red")
            return False
        
        if success:
            console.print(f"✅ {shell.title()} completion installed successfully!", style="bold green")
            console.print("🔄 Please restart your shell or run 'source ~/.bashrc' (bash) or 'source ~/.zshrc' (zsh)", style="blue")
        
        return success
    
    def uninstall_completion(self) -> bool:
        """Remove completion scripts."""
        try:
            # Remove completion directory
            if self.completion_dir.exists():
                import shutil
                shutil.rmtree(self.completion_dir)
            
            # Remove fish completion
            fish_completion = self.home / ".config" / "fish" / "completions" / "claude-tui.fish"
            if fish_completion.exists():
                fish_completion.unlink()
            
            # Remove zsh completion
            zsh_completion = self.home / ".zsh" / "completions" / "_claude-tui"
            if zsh_completion.exists():
                zsh_completion.unlink()
            
            console.print("✅ Completion scripts removed", style="green")
            console.print("💡 You may need to manually remove completion sources from your shell config files", style="blue")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to uninstall completion: {e}", style="red")
            return False
    
    def show_completion_status(self) -> None:
        """Show current completion installation status."""
        from rich.table import Table
        
        table = Table(title="Completion Status")
        table.add_column("Shell", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Location", style="blue")
        
        # Check bash completion
        bash_completion = self.completion_dir / "claude-tui-completion.bash"
        bash_status = "✅ Installed" if bash_completion.exists() else "❌ Not installed"
        table.add_row("Bash", bash_status, str(bash_completion))
        
        # Check fish completion
        fish_completion = self.home / ".config" / "fish" / "completions" / "claude-tui.fish"
        fish_status = "✅ Installed" if fish_completion.exists() else "❌ Not installed"
        table.add_row("Fish", fish_status, str(fish_completion))
        
        # Check zsh completion
        zsh_completion = self.home / ".zsh" / "completions" / "_claude-tui"
        zsh_status = "✅ Installed" if zsh_completion.exists() else "❌ Not installed"
        table.add_row("Zsh", zsh_status, str(zsh_completion))
        
        console.print(table)
        
        # Show current shell
        current_shell = self.detect_shell()
        console.print(f"\\n🐚 Current shell: {current_shell}", style="blue")


@click.group()
def completion() -> None:
    """Manage CLI completion system."""
    pass


@completion.command()
@click.option('--shell', type=click.Choice(['bash', 'fish', 'zsh', 'all']), help='Shell to install completion for')
@click.option('--force', is_flag=True, help='Force reinstallation')
def install(shell: Optional[str], force: bool) -> None:
    """
    Install CLI completion for your shell.
    
    Automatically detects your shell and installs appropriate
    completion scripts for enhanced CLI experience.
    
    Examples:
        claude-tui completion install
        claude-tui completion install --shell=bash
        claude-tui completion install --shell=all --force
    """
    manager = CompletionManager()
    
    if force:
        manager.uninstall_completion()
    
    success = manager.install_completion(shell)
    
    if success:
        console.print("\\n🎉 Completion installation complete!", style="bold green")
        console.print("\\n💡 Features enabled:", style="blue")
        console.print("  • Command and subcommand completion")
        console.print("  • Option name completion")
        console.print("  • Dynamic workspace and template completion")
        console.print("  • File and directory path completion")
        console.print("  • Context-aware suggestions")
    else:
        sys.exit(1)


@completion.command()
def uninstall() -> None:
    """
    Remove CLI completion scripts.
    
    Removes all installed completion scripts and configurations.
    """
    manager = CompletionManager()
    success = manager.uninstall_completion()
    
    if not success:
        sys.exit(1)


@completion.command()
def status() -> None:
    """
    Show completion installation status.
    
    Display which shells have completion installed and
    provide installation guidance.
    """
    manager = CompletionManager()
    manager.show_completion_status()


@completion.command()
def test() -> None:
    """
    Test completion functionality.
    
    Verify that completion is working correctly in your shell.
    """
    manager = CompletionManager()
    current_shell = manager.detect_shell()
    
    console.print(f"🧪 Testing completion for {current_shell}...", style="blue")
    
    # Check if completion is installed
    if current_shell == 'bash':
        completion_file = manager.completion_dir / "claude-tui-completion.bash"
        installed = completion_file.exists()
    elif current_shell == 'fish':
        completion_file = manager.home / ".config" / "fish" / "completions" / "claude-tui.fish"
        installed = completion_file.exists()
    elif current_shell == 'zsh':
        completion_file = manager.home / ".zsh" / "completions" / "_claude-tui"
        installed = completion_file.exists()
    else:
        console.print(f"❌ Unsupported shell: {current_shell}", style="red")
        return
    
    if installed:
        console.print("✅ Completion is installed", style="green")
        console.print("\\n🔧 Test completion by typing:", style="blue")
        console.print("  claude-tui <TAB><TAB>")
        console.print("  claude-tui core <TAB><TAB>")
        console.print("  claude-tui ai --<TAB><TAB>")
    else:
        console.print("❌ Completion is not installed", style="red")
        console.print("💡 Run 'claude-tui completion install' to install it", style="blue")


if __name__ == "__main__":
    completion()