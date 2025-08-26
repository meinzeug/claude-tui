#!/usr/bin/env python3
"""
Quick Production Client for Automatic Programming Pipeline

Provides a simple interface for using the automatic programming pipeline
to generate production-ready code from natural language requirements.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claude_tui.automation import AutomaticProgrammingCoordinator
from claude_tui.core.config_manager import ConfigManager


async def main():
    """Run a quick production client for the automatic programming pipeline."""
    
    print("\n" + "="*80)
    print("CLAUDE-TUI AUTOMATIC PROGRAMMING PIPELINE - PRODUCTION CLIENT")
    print("="*80)
    
    # Default production requirements example
    requirements = """
    Create a production-ready Python utility module that:
    1. Takes a list of numbers as input with type validation
    2. Calculates statistical measures (mean, median, std dev)
    3. Returns results with configurable precision
    4. Includes comprehensive error handling and logging
    5. Has complete docstring documentation with examples
    6. Includes comprehensive test suite with edge cases
    7. Follows PEP 8 and enterprise coding standards
    8. Includes performance benchmarks
    """
    
    print(f"\nRequirements:")
    print(requirements)
    
    # Create temporary directory for the generated project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = temp_dir
        
        print(f"\nProject Path: {project_path}")
        print("\nInitializing Production Programming Pipeline...")
        
        try:
            # Create and initialize the pipeline
            config_manager = ConfigManager()
            
            async with AutomaticProgrammingCoordinator(config_manager) as pipeline:
                print("Pipeline initialized successfully!")
                
                print("\nGenerating production code from requirements...")
                
                # Generate production code
                result = await pipeline.generate_code(
                    requirements=requirements,
                    project_path=project_path,
                    options={
                        'production_mode': True,
                        'include_tests': True,
                        'include_benchmarks': True,
                        'enterprise_standards': True
                    }
                )
                
                print("\n" + "-"*60)
                print("GENERATION RESULTS")
                print("-"*60)
                
                print(f"Success: {result.success}")
                print(f"Execution Time: {result.execution_time:.2f}s")
                print(f"Quality Score: {result.quality_score:.2f}/1.0")
                print(f"Generated Files: {len(result.generated_files)}")
                
                if result.requirements_analysis:
                    analysis = result.requirements_analysis
                    print(f"\nProject Analysis:")
                    print(f"  Type: {analysis.project_type}")
                    print(f"  Complexity: {analysis.estimated_complexity}/10")
                    print(f"  Architecture: {analysis.recommended_architecture}")
                    print(f"  Technologies: {', '.join(analysis.suggested_technologies)}")
                
                if result.task_components:
                    print(f"\nTask Components ({len(result.task_components)}):")
                    for task in result.task_components:
                        print(f"  - {task.id}: {task.description}")
                        print(f"    Agent: {task.agent_type}, Priority: {task.priority}")
                
                if result.generated_files:
                    print(f"\nGenerated Files:")
                    for file_path in result.generated_files:
                        print(f"  - {file_path}")
                        
                        # Show file content if it exists
                        full_path = Path(project_path) / file_path
                        if full_path.exists() and full_path.stat().st_size < 2000:  # Show small files only
                            print(f"    Content preview:")
                            try:
                                with open(full_path, 'r') as f:
                                    content = f.read()
                                    # Show first few lines
                                    lines = content.split('\n')[:10]
                                    for line in lines:
                                        print(f"      {line}")
                                    if len(content.split('\n')) > 10:
                                        print(f"      ... ({len(content.split('\n')) - 10} more lines)")
                            except Exception as e:
                                print(f"      Error reading file: {e}")
                
                if result.validation_results:
                    validation = result.validation_results
                    print(f"\nValidation:")
                    print(f"  Overall Success: {validation.get('overall_success', False)}")
                    print(f"  Quality Score: {validation.get('quality_score', 0):.2f}")
                    
                    issues = validation.get('issues', [])
                    if issues:
                        print(f"  Issues Found: {len(issues)}")
                        for issue in issues[:3]:  # Show first 3 issues
                            print(f"    - {issue}")
                    else:
                        print("  No validation issues found!")
                
                if result.agent_reports:
                    print(f"\nAgent Reports ({len(result.agent_reports)}):")
                    for agent_id, report in result.agent_reports.items():
                        success_indicator = "✓" if report.get('success', False) else "✗"
                        print(f"  {success_indicator} {agent_id} ({report.get('agent_type', 'unknown')})")
                        
                        if report.get('execution_time'):
                            print(f"    Execution time: {report['execution_time']:.2f}s")
                
                if result.error_message:
                    print(f"\nError: {result.error_message}")
                
                # Demo memory context
                print(f"\nProduction Context:")
                memory_context = await pipeline.get_memory_context('pipeline_context')
                if memory_context:
                    print(f"  Pipeline ID: {memory_context.get('pipeline_id', 'N/A')}")
                    print(f"  Start Time: {memory_context.get('start_time', 'N/A')}")
                    print(f"  Environment: Production")
                    print(f"  Quality Assurance: Enabled")
                else:
                    print("  No production context available")
                
                print("\n" + "="*80)
                if result.success:
                    print("✅ PRODUCTION CODE GENERATION COMPLETED SUCCESSFULLY!")
                else:
                    print("❌ PRODUCTION GENERATION COMPLETED WITH ERRORS")
                print("="*80)
                
                return result
        
        except Exception as e:
            print(f"\n❌ Production generation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Run the production client
    result = asyncio.run(main())
    
    # Exit with appropriate code
    if result and result.success:
        sys.exit(0)
    else:
        sys.exit(1)