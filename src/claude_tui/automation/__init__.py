"""
Automatic Programming Automation Module

This module provides automatic programming capabilities that transform natural language
requirements into production-ready code using Claude Code direct CLI integration
and Claude Flow orchestration.

Key Components:
- AutomaticProgrammingCoordinator: Main pipeline orchestrator
- RequirementsAnalyzer: Natural language requirements parser
- TaskDecomposer: Complex task breakdown with dependency management
- CodeGenerator: Multi-agent coordinated code generation
- ValidationEngine: Code quality and requirements validation

Usage:
    from src.claude_tui.automation import generate_project_from_requirements
    
    result = await generate_project_from_requirements(
        requirements="Create a REST API with user authentication",
        project_path="/path/to/project"
    )
"""

from .automatic_programming import (
    AutomaticProgrammingCoordinator,
    RequirementsAnalyzer,
    TaskDecomposer,
    CodeGenerator,
    ValidationEngine,
    PipelineResult,
    PipelineStatus,
    TaskComponent,
    RequirementsAnalysis,
    create_programming_pipeline,
    generate_project_from_requirements
)

__all__ = [
    'AutomaticProgrammingCoordinator',
    'RequirementsAnalyzer', 
    'TaskDecomposer',
    'CodeGenerator',
    'ValidationEngine',
    'PipelineResult',
    'PipelineStatus',
    'TaskComponent',
    'RequirementsAnalysis',
    'create_programming_pipeline',
    'generate_project_from_requirements'
]

__version__ = "1.0.0"