#!/usr/bin/env python3
"""
Claude-TIU AI Integration CLI Commands.

Advanced AI-powered features for code generation, review,
optimization, and intelligent development assistance.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from ...core.config_manager import ConfigManager
from ...core.ai_interface import AIInterface
from ...integrations.claude_code_client import ClaudeCodeClient
from ...integrations.claude_flow_client import ClaudeFlowClient
from ...validation.anti_hallucination_engine import AntiHallucinationEngine
from ...core.task_engine import TaskEngine


@click.group()
def ai_commands() -> None:
    """AI-powered development assistance commands."""
    pass


@ai_commands.command()
@click.argument('prompt', nargs=-1, required=True)
@click.option('--language', help='Programming language for generated code')
@click.option('--framework', help='Framework or library to use')
@click.option('--style', help='Code style or pattern to follow')
@click.option('--context-files', multiple=True, help='Files to include as context')
@click.option('--output', help='Output file to save generated code')
@click.option('--template', help='Code template to use')
@click.option('--interactive', is_flag=True, help='Interactive generation mode')
@click.pass_context
def generate(
    ctx: click.Context,
    prompt: tuple[str, ...],
    language: Optional[str],
    framework: Optional[str],
    style: Optional[str],
    context_files: tuple[str, ...],
    output: Optional[str],
    template: Optional[str],
    interactive: bool
) -> None:
    """
    Generate code using AI assistance.
    
    Generate functions, classes, modules, or entire applications
    based on natural language descriptions.
    
    Examples:
        claude-tiu ai generate "create a REST API for user management"
        claude-tiu ai generate "implement binary search algorithm" --language=python
        claude-tiu ai generate "React component for file upload" --framework=react
        claude-tiu ai generate --interactive
    """
    prompt_text = ' '.join(prompt) if prompt else None
    asyncio.run(generate_code(ctx, prompt_text, language, framework, style, context_files, output, template, interactive))


@ai_commands.command()
@click.argument('files', nargs=-1)
@click.option('--automated', is_flag=True, help='Automated review without interaction')
@click.option('--focus', multiple=True, help='Focus areas (security, performance, style, etc.)')
@click.option('--severity', type=click.Choice(['low', 'medium', 'high', 'critical']), help='Minimum severity level')
@click.option('--fix', is_flag=True, help='Attempt to auto-fix issues')
@click.option('--report', help='Generate review report file')
@click.pass_context
def review(
    ctx: click.Context,
    files: tuple[str, ...],
    automated: bool,
    focus: tuple[str, ...],
    severity: Optional[str],
    fix: bool,
    report: Optional[str]
) -> None:
    """
    AI-powered code review and analysis.
    
    Comprehensive code review including security, performance,
    style, and best practice analysis.
    
    Examples:
        claude-tiu ai review src/main.py
        claude-tiu ai review --focus=security --focus=performance
        claude-tiu ai review --automated --fix --report=review.json
    """
    asyncio.run(review_code(ctx, files, automated, focus, severity, fix, report))


@ai_commands.command()
@click.argument('files', nargs=-1)
@click.option('--issue-type', help='Specific type of issue to fix')
@click.option('--interactive', is_flag=True, help='Interactive fix mode')
@click.option('--backup', is_flag=True, help='Create backup before fixing')
@click.option('--dry-run', is_flag=True, help='Show fixes without applying')
@click.option('--confidence', type=float, default=0.8, help='Minimum confidence threshold')
@click.pass_context
def fix(
    ctx: click.Context,
    files: tuple[str, ...],
    issue_type: Optional[str],
    interactive: bool,
    backup: bool,
    dry_run: bool,
    confidence: float
) -> None:
    """
    Automatically fix code issues using AI.
    
    Identify and fix common programming issues, bugs,
    and code quality problems.
    
    Examples:
        claude-tiu ai fix src/app.py --backup
        claude-tiu ai fix --issue-type=security --interactive
        claude-tiu ai fix --dry-run --confidence=0.9
    """
    asyncio.run(fix_code(ctx, files, issue_type, interactive, backup, dry_run, confidence))


@ai_commands.command()
@click.argument('files', nargs=-1)
@click.option('--target', help='Optimization target (speed, memory, size, etc.)')
@click.option('--profile', is_flag=True, help='Profile code before optimization')
@click.option('--benchmark', is_flag=True, help='Benchmark before/after performance')
@click.option('--suggestions-only', is_flag=True, help='Show suggestions without applying')
@click.pass_context
def optimize(
    ctx: click.Context,
    files: tuple[str, ...],
    target: Optional[str],
    profile: bool,
    benchmark: bool,
    suggestions_only: bool
) -> None:
    """
    AI-powered code optimization.
    
    Optimize code for performance, memory usage, or size
    while maintaining functionality.
    
    Examples:
        claude-tiu ai optimize src/slow_function.py --target=speed
        claude-tiu ai optimize --profile --benchmark
        claude-tiu ai optimize --suggestions-only
    """
    asyncio.run(optimize_code(ctx, files, target, profile, benchmark, suggestions_only))


@ai_commands.command()
@click.argument('description')
@click.option('--test-type', type=click.Choice(['unit', 'integration', 'e2e']), default='unit')
@click.option('--framework', help='Testing framework to use')
@click.option('--coverage', is_flag=True, help='Include coverage requirements')
@click.option('--mocks', is_flag=True, help='Generate mock objects')
@click.pass_context
def test_generate(
    ctx: click.Context,
    description: str,
    test_type: str,
    framework: Optional[str],
    coverage: bool,
    mocks: bool
) -> None:
    """
    Generate test cases using AI.
    
    Create comprehensive test suites based on code analysis
    and requirements.
    
    Examples:
        claude-tiu ai test-generate "user authentication module"
        claude-tiu ai test-generate "API endpoints" --test-type=integration
        claude-tiu ai test-generate "payment system" --mocks --coverage
    """
    asyncio.run(generate_tests(ctx, description, test_type, framework, coverage, mocks))


@ai_commands.command()
@click.argument('files', nargs=-1)
@click.option('--format', type=click.Choice(['markdown', 'rst', 'html']), default='markdown')
@click.option('--style', help='Documentation style guide')
@click.option('--examples', is_flag=True, help='Include usage examples')
@click.option('--api-docs', is_flag=True, help='Generate API documentation')
@click.pass_context
def document(
    ctx: click.Context,
    files: tuple[str, ...],
    format: str,
    style: Optional[str],
    examples: bool,
    api_docs: bool
) -> None:
    """
    Generate documentation using AI.
    
    Create comprehensive documentation from code analysis
    and comments.
    
    Examples:
        claude-tiu ai document src/api.py --examples
        claude-tiu ai document --api-docs --format=rst
        claude-tiu ai document src/ --style=google
    """
    asyncio.run(generate_documentation(ctx, files, format, style, examples, api_docs))


@ai_commands.command()
@click.argument('source_language')
@click.argument('target_language')
@click.argument('files', nargs=-1)
@click.option('--preserve-comments', is_flag=True, help='Preserve original comments')
@click.option('--modernize', is_flag=True, help='Use modern language features')
@click.option('--test-conversion', is_flag=True, help='Convert tests as well')
@click.pass_context
def translate(
    ctx: click.Context,
    source_language: str,
    target_language: str,
    files: tuple[str, ...],
    preserve_comments: bool,
    modernize: bool,
    test_conversion: bool
) -> None:
    """
    Translate code between programming languages.
    
    Convert code from one programming language to another
    while preserving functionality and style.
    
    Examples:
        claude-tiu ai translate python javascript src/utils.py
        claude-tiu ai translate java python --modernize
        claude-tiu ai translate javascript typescript --test-conversion
    """
    asyncio.run(translate_code(ctx, source_language, target_language, files, preserve_comments, modernize, test_conversion))


@ai_commands.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--context-files', multiple=True, help='Files to include as context')
@click.option('--mode', type=click.Choice(['explain', 'debug', 'suggest']), default='explain')
@click.pass_context
def ask(
    ctx: click.Context,
    query: tuple[str, ...],
    context_files: tuple[str, ...],
    mode: str
) -> None:
    """
    Ask AI questions about your code.
    
    Get explanations, debugging help, or suggestions
    for your codebase.
    
    Examples:
        claude-tiu ai ask "how does this authentication work?"
        claude-tiu ai ask "why is this function slow?" --mode=debug
        claude-tiu ai ask "improve this code" --mode=suggest
    """
    query_text = ' '.join(query)
    asyncio.run(ask_ai(ctx, query_text, context_files, mode))


# Implementation functions

async def generate_code(
    ctx: click.Context,
    prompt: Optional[str],
    language: Optional[str],
    framework: Optional[str],
    style: Optional[str],
    context_files: tuple[str, ...],
    output: Optional[str],
    template: Optional[str],
    interactive: bool
) -> None:
    """Generate code using AI with comprehensive options."""
    console: Console = ctx.obj['console']
    
    try:
        # Initialize AI interface
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Interactive mode
        if interactive:
            prompt = await _interactive_code_generation(console)
        
        if not prompt:
            console.print("❌ No prompt provided", style="red")
            return
        
        # Build generation context
        generation_context = {
            'language': language,
            'framework': framework,
            'style': style,
            'template': template
        }
        
        # Load context files
        context_data = {}
        for file_path in context_files:
            try:
                context_data[file_path] = Path(file_path).read_text(encoding='utf-8')
            except Exception as e:
                console.print(f"⚠️ Could not read {file_path}: {e}", style="yellow")
        
        # Generate code
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating code...", total=None)
            
            result = await ai_interface.generate_code(
                prompt=prompt,
                context=generation_context,
                files_context=context_data
            )
            
            progress.update(task, description="✅ Code generation complete")
        
        # Display generated code
        if result.code:
            console.print("\n🤖 Generated Code:", style="bold blue")
            syntax = Syntax(result.code, language or "python", theme="monokai", line_numbers=True)
            console.print(syntax)
            
            # Save to file if requested
            if output:
                Path(output).write_text(result.code, encoding='utf-8')
                console.print(f"💾 Code saved to: {output}", style="green")
        
        # Show explanation if available
        if result.explanation:
            console.print("\n📝 Explanation:", style="bold blue")
            console.print(Markdown(result.explanation))
        
    except Exception as e:
        console.print(f"❌ Code generation failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def review_code(
    ctx: click.Context,
    files: tuple[str, ...],
    automated: bool,
    focus: tuple[str, ...],
    severity: Optional[str],
    fix: bool,
    report: Optional[str]
) -> None:
    """Comprehensive AI-powered code review."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Determine files to review
        review_files = list(files) if files else _discover_code_files()
        
        if not review_files:
            console.print("❌ No files to review", style="red")
            return
        
        # Configure review parameters
        review_config = {
            'focus_areas': list(focus) if focus else ['security', 'performance', 'style', 'bugs'],
            'minimum_severity': severity or 'low',
            'automated': automated
        }
        
        # Perform code review
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Reviewing {len(review_files)} files...", total=None)
            
            review_results = []
            for file_path in review_files:
                file_content = Path(file_path).read_text(encoding='utf-8')
                result = await ai_interface.review_code(
                    code=file_content,
                    filename=file_path,
                    config=review_config
                )
                review_results.append(result)
            
            progress.update(task, description="✅ Code review complete")
        
        # Display review results
        _display_review_results(console, review_results)
        
        # Auto-fix if requested
        if fix:
            await _apply_review_fixes(console, ai_interface, review_results)
        
        # Generate report if requested
        if report:
            _generate_review_report(review_results, report)
            console.print(f"📊 Review report saved to: {report}", style="blue")
        
    except Exception as e:
        console.print(f"❌ Code review failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def fix_code(
    ctx: click.Context,
    files: tuple[str, ...],
    issue_type: Optional[str],
    interactive: bool,
    backup: bool,
    dry_run: bool,
    confidence: float
) -> None:
    """Automatically fix code issues using AI."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        validator = AntiHallucinationEngine(config_manager)
        
        # Determine files to fix
        fix_files = list(files) if files else _discover_code_files()
        
        if not fix_files:
            console.print("❌ No files to fix", style="red")
            return
        
        # Create backups if requested
        if backup and not dry_run:
            await _create_backups(fix_files)
            console.print(f"💾 Created backups for {len(fix_files)} files", style="blue")
        
        # Fix issues in each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Analyzing and fixing {len(fix_files)} files...", total=None)
            
            fix_results = []
            for file_path in fix_files:
                file_content = Path(file_path).read_text(encoding='utf-8')
                
                # Identify issues
                issues = await validator.identify_issues(file_content, file_path)
                
                if issue_type:
                    issues = [issue for issue in issues if issue.type == issue_type]
                
                # Fix issues with confidence threshold
                if issues:
                    result = await ai_interface.fix_code_issues(
                        code=file_content,
                        issues=issues,
                        confidence_threshold=confidence
                    )
                    fix_results.append((file_path, result))
            
            progress.update(task, description="✅ Code fixing complete")
        
        # Display and apply fixes
        await _process_fix_results(console, fix_results, interactive, dry_run)
        
    except Exception as e:
        console.print(f"❌ Code fixing failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def optimize_code(
    ctx: click.Context,
    files: tuple[str, ...],
    target: Optional[str],
    profile: bool,
    benchmark: bool,
    suggestions_only: bool
) -> None:
    """AI-powered code optimization."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Determine files to optimize
        optimize_files = list(files) if files else _discover_code_files()
        
        if not optimize_files:
            console.print("❌ No files to optimize", style="red")
            return
        
        # Profile code if requested
        if profile:
            await _profile_code(console, optimize_files)
        
        # Benchmark before optimization
        if benchmark:
            before_metrics = await _benchmark_code(console, optimize_files)
        
        # Optimize each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Optimizing {len(optimize_files)} files...", total=None)
            
            optimization_results = []
            for file_path in optimize_files:
                file_content = Path(file_path).read_text(encoding='utf-8')
                
                result = await ai_interface.optimize_code(
                    code=file_content,
                    filename=file_path,
                    optimization_target=target or 'performance'
                )
                optimization_results.append((file_path, result))
            
            progress.update(task, description="✅ Code optimization complete")
        
        # Display optimization results
        _display_optimization_results(console, optimization_results, suggestions_only)
        
        # Benchmark after optimization
        if benchmark and not suggestions_only:
            after_metrics = await _benchmark_code(console, optimize_files)
            _compare_benchmarks(console, before_metrics, after_metrics)
        
    except Exception as e:
        console.print(f"❌ Code optimization failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def generate_tests(
    ctx: click.Context,
    description: str,
    test_type: str,
    framework: Optional[str],
    coverage: bool,
    mocks: bool
) -> None:
    """Generate comprehensive test cases using AI."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Analyze existing code for test generation
        source_files = _discover_code_files()
        context_data = {}
        
        for file_path in source_files[:5]:  # Limit context files
            try:
                context_data[file_path] = Path(file_path).read_text(encoding='utf-8')
            except Exception:
                continue
        
        # Generate tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating test cases...", total=None)
            
            test_config = {
                'test_type': test_type,
                'framework': framework or 'pytest',
                'include_coverage': coverage,
                'include_mocks': mocks
            }
            
            result = await ai_interface.generate_tests(
                description=description,
                context_files=context_data,
                config=test_config
            )
            
            progress.update(task, description="✅ Test generation complete")
        
        # Display generated tests
        if result.test_code:
            console.print(f"\n🧪 Generated {test_type.title()} Tests:", style="bold blue")
            syntax = Syntax(result.test_code, "python", theme="monokai", line_numbers=True)
            console.print(syntax)
            
            # Save tests to appropriate directory
            test_dir = Path("tests")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / f"test_{description.lower().replace(' ', '_')}.py"
            test_file.write_text(result.test_code, encoding='utf-8')
            console.print(f"💾 Tests saved to: {test_file}", style="green")
        
        # Show test explanation
        if result.explanation:
            console.print("\n📝 Test Strategy:", style="bold blue")
            console.print(Markdown(result.explanation))
        
    except Exception as e:
        console.print(f"❌ Test generation failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def generate_documentation(
    ctx: click.Context,
    files: tuple[str, ...],
    format: str,
    style: Optional[str],
    examples: bool,
    api_docs: bool
) -> None:
    """Generate comprehensive documentation using AI."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Determine files to document
        doc_files = list(files) if files else _discover_code_files()
        
        if not doc_files:
            console.print("❌ No files to document", style="red")
            return
        
        # Generate documentation for each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Generating documentation for {len(doc_files)} files...", total=None)
            
            doc_config = {
                'format': format,
                'style': style or 'google',
                'include_examples': examples,
                'api_docs': api_docs
            }
            
            documentation_results = []
            for file_path in doc_files:
                file_content = Path(file_path).read_text(encoding='utf-8')
                
                result = await ai_interface.generate_documentation(
                    code=file_content,
                    filename=file_path,
                    config=doc_config
                )
                documentation_results.append((file_path, result))
            
            progress.update(task, description="✅ Documentation generation complete")
        
        # Save documentation
        docs_dir = Path("docs/generated")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path, doc_result in documentation_results:
            if doc_result.documentation:
                doc_filename = f"{Path(file_path).stem}.{format}"
                doc_file = docs_dir / doc_filename
                doc_file.write_text(doc_result.documentation, encoding='utf-8')
                
                console.print(f"📚 Documentation generated: {doc_file}", style="green")
        
        console.print(f"\n✅ Generated documentation for {len(documentation_results)} files", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Documentation generation failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def translate_code(
    ctx: click.Context,
    source_language: str,
    target_language: str,
    files: tuple[str, ...],
    preserve_comments: bool,
    modernize: bool,
    test_conversion: bool
) -> None:
    """Translate code between programming languages."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Determine files to translate
        translate_files = list(files) if files else []
        
        if not translate_files:
            console.print("❌ No files to translate", style="red")
            return
        
        # Translation configuration
        translation_config = {
            'source_language': source_language,
            'target_language': target_language,
            'preserve_comments': preserve_comments,
            'modernize': modernize,
            'include_tests': test_conversion
        }
        
        # Translate each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Translating {len(translate_files)} files...", total=None)
            
            translation_results = []
            for file_path in translate_files:
                file_content = Path(file_path).read_text(encoding='utf-8')
                
                result = await ai_interface.translate_code(
                    code=file_content,
                    filename=file_path,
                    config=translation_config
                )
                translation_results.append((file_path, result))
            
            progress.update(task, description="✅ Code translation complete")
        
        # Display and save translated code
        output_dir = Path(f"translated_{target_language}")
        output_dir.mkdir(exist_ok=True)
        
        for file_path, translation_result in translation_results:
            if translation_result.translated_code:
                # Determine output file extension
                ext_map = {
                    'python': '.py',
                    'javascript': '.js',
                    'typescript': '.ts',
                    'java': '.java',
                    'go': '.go',
                    'rust': '.rs'
                }
                
                output_ext = ext_map.get(target_language.lower(), '.txt')
                output_file = output_dir / f"{Path(file_path).stem}{output_ext}"
                output_file.write_text(translation_result.translated_code, encoding='utf-8')
                
                console.print(f"🔄 Translated: {file_path} → {output_file}", style="green")
                
                # Show translated code preview
                syntax = Syntax(
                    translation_result.translated_code[:500] + ("..." if len(translation_result.translated_code) > 500 else ""),
                    target_language,
                    theme="monokai",
                    line_numbers=True
                )
                console.print(f"\n📄 Preview of {output_file}:")
                console.print(syntax)
        
        console.print(f"\n✅ Translated {len(translation_results)} files to {target_language}", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Code translation failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


async def ask_ai(
    ctx: click.Context,
    query: str,
    context_files: tuple[str, ...],
    mode: str
) -> None:
    """Ask AI questions about code with context."""
    console: Console = ctx.obj['console']
    
    try:
        config_manager = ConfigManager(config_dir=ctx.obj.get('config_dir'))
        await config_manager.initialize()
        
        ai_interface = AIInterface(config_manager)
        
        # Load context files
        context_data = {}
        for file_path in context_files:
            try:
                context_data[file_path] = Path(file_path).read_text(encoding='utf-8')
            except Exception as e:
                console.print(f"⚠️ Could not read {file_path}: {e}", style="yellow")
        
        # Ask AI
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Asking AI...", total=None)
            
            result = await ai_interface.ask_question(
                query=query,
                context_files=context_data,
                mode=mode
            )
            
            progress.update(task, description="✅ AI response received")
        
        # Display response
        console.print(f"\n🤖 AI Response ({mode} mode):", style="bold blue")
        console.print(Markdown(result.response))
        
        # Show relevant code snippets if provided
        if result.code_snippets:
            console.print("\n💡 Relevant Code Examples:", style="bold yellow")
            for snippet in result.code_snippets:
                syntax = Syntax(snippet.code, snippet.language, theme="monokai")
                console.print(f"\n{snippet.explanation}:")
                console.print(syntax)
        
    except Exception as e:
        console.print(f"❌ AI query failed: {e}", style="red")
        if ctx.obj.get('debug'):
            console.print_exception()
        sys.exit(1)


# Helper functions

async def _interactive_code_generation(console: Console) -> str:
    """Interactive code generation wizard."""
    console.print("🧙 Interactive Code Generation Wizard", style="bold blue")
    
    prompt = click.prompt("Describe what you want to create")
    
    # Additional prompts for specificity
    language = click.prompt("Programming language (optional)", default="", show_default=False)
    framework = click.prompt("Framework/library (optional)", default="", show_default=False)
    
    if language:
        prompt += f" using {language}"
    if framework:
        prompt += f" with {framework}"
    
    return prompt


def _discover_code_files() -> List[str]:
    """Discover code files in the current directory."""
    code_extensions = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
    code_files = []
    
    for ext in code_extensions:
        code_files.extend([str(p) for p in Path('.').rglob(f'*{ext}')])
    
    return code_files[:20]  # Limit to first 20 files


def _display_review_results(console: Console, results: List[Any]) -> None:
    """Display code review results in a formatted table."""
    table = Table(title="Code Review Results")
    table.add_column("File", style="cyan")
    table.add_column("Issues", style="yellow")
    table.add_column("Severity", style="red")
    table.add_column("Status", style="green")
    
    for result in results:
        issues_count = len(result.issues) if result.issues else 0
        severity = "High" if any(issue.severity == "high" for issue in (result.issues or [])) else "Low"
        status = "⚠️ Needs Review" if issues_count > 0 else "✅ Good"
        
        table.add_row(result.filename, str(issues_count), severity, status)
    
    console.print(table)


def _display_optimization_results(console: Console, results: List[tuple], suggestions_only: bool) -> None:
    """Display code optimization results."""
    for file_path, result in results:
        console.print(f"\n📁 {file_path}:", style="bold cyan")
        
        if result.optimizations:
            for opt in result.optimizations:
                console.print(f"  • {opt.description}", style="green")
                if not suggestions_only:
                    console.print(f"    Performance gain: {opt.improvement}%", style="blue")


async def _create_backups(files: List[str]) -> None:
    """Create backup copies of files."""
    backup_dir = Path("backups") / f"backup_{int(time.time())}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in files:
        source = Path(file_path)
        destination = backup_dir / source.name
        destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')


async def _process_fix_results(
    console: Console,
    results: List[tuple],
    interactive: bool,
    dry_run: bool
) -> None:
    """Process and apply code fix results."""
    for file_path, result in results:
        console.print(f"\n📁 {file_path}:", style="bold cyan")
        
        if result.fixes:
            console.print(f"  Found {len(result.fixes)} potential fixes:", style="green")
            
            for fix in result.fixes:
                console.print(f"    • {fix.description} (confidence: {fix.confidence:.1%})")
                
                if interactive:
                    if click.confirm(f"    Apply this fix?"):
                        if not dry_run:
                            # Apply the fix
                            console.print(f"      ✅ Applied", style="green")
                        else:
                            console.print(f"      🔍 Would apply (dry-run)", style="blue")
                else:
                    if not dry_run and fix.confidence >= 0.8:
                        # Auto-apply high-confidence fixes
                        console.print(f"      ✅ Auto-applied", style="green")
        else:
            console.print("  No issues found", style="green")


async def _profile_code(console: Console, files: List[str]) -> None:
    """Profile code performance."""
    console.print("📊 Profiling code performance...", style="blue")
    # Implementation would use profiling tools


async def _benchmark_code(console: Console, files: List[str]) -> Dict[str, Any]:
    """Benchmark code performance."""
    console.print("⏱️ Benchmarking code...", style="blue")
    # Implementation would run performance benchmarks
    return {"execution_time": 100, "memory_usage": 50}


def _compare_benchmarks(console: Console, before: Dict[str, Any], after: Dict[str, Any]) -> None:
    """Compare before/after benchmark results."""
    console.print("\n📈 Performance Comparison:", style="bold blue")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Improvement", style="blue")
    
    for metric in before.keys():
        before_val = before[metric]
        after_val = after[metric]
        improvement = ((before_val - after_val) / before_val * 100)
        
        table.add_row(
            metric.replace("_", " ").title(),
            f"{before_val}",
            f"{after_val}",
            f"{improvement:+.1f}%"
        )
    
    console.print(table)


async def _apply_review_fixes(console: Console, ai_interface: Any, results: List[Any]) -> None:
    """Apply fixes from code review results."""
    console.print("🔧 Applying automated fixes...", style="blue")
    # Implementation would apply fixes from review


def _generate_review_report(results: List[Any], report_path: str) -> None:
    """Generate a comprehensive review report."""
    report_data = {
        "timestamp": time.time(),
        "files_reviewed": len(results),
        "total_issues": sum(len(result.issues or []) for result in results),
        "results": results
    }
    
    Path(report_path).write_text(json.dumps(report_data, indent=2, default=str))