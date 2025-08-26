#!/usr/bin/env python3
"""
Automatic Programming Pipeline Production Suite

Production-ready automatic programming pipeline that generates
complete, deployment-ready projects from natural language requirements.

Usage:
    python scripts/production_programming_suite.py [--template <name>] [--output <path>]
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claude_tui.automation import generate_project_from_requirements
from claude_tui.core.config_manager import ConfigManager
from claude_tui.core.logger import Logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = Logger(__name__)


async def generate_production_rest_api():
    """Generate a production-ready REST API from requirements."""
    print("\n" + "="*80)
    print("PRODUCTION CODE GENERATION: REST API")
    print("="*80)
    
    requirements = """
    Create a production-ready FastAPI REST API for enterprise task management:
    
    1. Enterprise User Management:
       - Multi-factor authentication
       - Role-based access control (RBAC)
       - User profile management with audit trails
       - Session management and security
    
    2. Advanced Task Management:
       - Full CRUD operations with soft delete
       - Task hierarchies and dependencies
       - Advanced filtering and search
       - Real-time updates via WebSocket
       - Task templates and automation
    
    3. Production Requirements:
       - PostgreSQL with connection pooling
       - Redis caching layer
       - Comprehensive input validation
       - Enterprise error handling
       - OpenAPI documentation with examples
       - Docker multi-stage builds
       - Kubernetes deployment manifests
       - Comprehensive test coverage (>95%)
       - Performance monitoring
       - Security scanning integration
    
    4. Enterprise Standards:
       - Clean architecture patterns
       - Comprehensive logging and tracing
       - Security hardening (OWASP)
       - Performance optimization
       - Automated quality gates
    """
    
    project_path = "/tmp/production_task_management_api"
    
    print(f"Requirements: {requirements[:200]}...")
    print(f"Target Path: {project_path}")
    print(f"Starting generation at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Generate production project
        result = await generate_project_from_requirements(
            requirements=requirements,
            project_path=project_path,
            options={
                'production_mode': True,
                'include_docker': True,
                'include_kubernetes': True,
                'include_tests': True,
                'include_security_scan': True,
                'preferred_framework': 'FastAPI',
                'database': 'PostgreSQL',
                'cache': 'Redis',
                'orm': 'SQLAlchemy',
                'monitoring': 'Prometheus',
                'logging': 'Structured'
            }
        )
        
        # Display results
        print("\n" + "-"*60)
        print("GENERATION RESULTS")
        print("-"*60)
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        print(f"Quality Score: {result.quality_score:.2f}/1.0")
        print(f"Generated Files: {len(result.generated_files)}")
        
        if result.requirements_analysis:
            print(f"Project Type: {result.requirements_analysis.project_type}")
            print(f"Architecture: {result.requirements_analysis.recommended_architecture}")
            print(f"Technologies: {', '.join(result.requirements_analysis.suggested_technologies)}")
        
        if result.generated_files:
            print("\nGenerated Files:")
            for file_path in result.generated_files:
                print(f"  - {file_path}")
        
        if result.validation_results:
            validation = result.validation_results
            print(f"\nValidation Results:")
            print(f"  Overall Success: {validation.get('overall_success', False)}")
            print(f"  Requirements Coverage: {validation.get('requirements_coverage', 0):.1f}%")
            print(f"  Issues Found: {len(validation.get('issues', []))}")
            print(f"  Suggestions: {len(validation.get('suggestions', []))}")
        
        if result.task_components:
            print(f"\nTask Components: {len(result.task_components)}")
            for task in result.task_components:
                print(f"  - {task.id}: {task.description[:50]}...")
        
        if result.agent_reports:
            print(f"\nAgent Reports: {len(result.agent_reports)} agents")
            for agent_id, report in result.agent_reports.items():
                status = "✓" if report.get('success', False) else "✗"
                print(f"  {status} {agent_id}: {report.get('agent_type', 'unknown')}")
        
        if result.error_message:
            print(f"\nError: {result.error_message}")
        
        # Save detailed results
        results_file = Path(project_path) / "generation_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            # Convert dataclasses to dicts for JSON serialization
            json_result = {
                'success': result.success,
                'execution_time': result.execution_time,
                'quality_score': result.quality_score,
                'generated_files': result.generated_files,
                'validation_results': result.validation_results,
                'error_message': result.error_message,
                'agent_reports': result.agent_reports,
                'requirements_analysis': {
                    'project_type': result.requirements_analysis.project_type,
                    'estimated_complexity': result.requirements_analysis.estimated_complexity,
                    'recommended_architecture': result.requirements_analysis.recommended_architecture,
                    'suggested_technologies': result.requirements_analysis.suggested_technologies
                } if result.requirements_analysis else None,
                'task_components': [
                    {
                        'id': task.id,
                        'description': task.description,
                        'file_path': task.file_path,
                        'agent_type': task.agent_type,
                        'priority': task.priority,
                        'estimated_complexity': task.estimated_complexity
                    }
                    for task in result.task_components
                ]
            }
            
            json.dump(json_result, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return result
        
    except Exception as e:
        print(f"\nGeneration failed with error: {e}")
        logger.error(f"Demo generation failed: {e}", exc_info=True)
        return None


async def generate_production_utility():
    """Generate a production-ready Python utility."""
    print("\n" + "="*80)
    print("PRODUCTION CODE GENERATION: Utility Module")
    print("="*80)
    
    requirements = """
    Create a production-ready Python data processing utility:
    1. Reads CSV files with configurable schema validation
    2. Validates data using comprehensive rules (email, phone, etc.)
    3. Applies configurable filters with complex conditions
    4. Exports results in multiple formats (CSV, JSON, Excel)
    5. Includes enterprise-grade error handling and structured logging
    6. Has comprehensive CLI with auto-completion
    7. Includes full test suite with performance benchmarks
    8. Supports parallel processing for large datasets
    9. Includes data quality reporting
    10. Enterprise security and audit logging
    """
    
    project_path = "/tmp/production_data_processor"
    
    print(f"Requirements: {requirements}")
    print(f"Target Path: {project_path}")
    
    try:
        result = await generate_project_from_requirements(
            requirements=requirements,
            project_path=project_path,
            options={
                'production_mode': True,
                'include_tests': True,
                'include_benchmarks': True,
                'script_type': 'enterprise_utility',
                'cli_framework': 'click',
                'parallel_processing': True,
                'security_validation': True
            }
        )
        
        print(f"\nProduction Utility Generation Result:")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Files: {len(result.generated_files)}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"Production utility generation failed: {e}")
        return None


async def run_production_benchmark():
    """Run production benchmark on the automatic programming pipeline."""
    print("\n" + "="*80)
    print("PRODUCTION PROGRAMMING BENCHMARK")
    print("="*80)
    
    test_cases = [
        {
            'name': 'Simple Function',
            'requirements': 'Create a Python function that calculates fibonacci numbers with memoization',
            'expected_files': 1
        },
        {
            'name': 'CLI Tool', 
            'requirements': 'Create a command-line calculator that supports basic math operations with argument parsing',
            'expected_files': 2
        },
        {
            'name': 'Web Scraper',
            'requirements': 'Create a web scraper that extracts article titles from a news website and saves to JSON',
            'expected_files': 3
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nBenchmark {i+1}/{len(test_cases)}: {test_case['name']}")
        
        start_time = time.time()
        
        try:
            result = await generate_project_from_requirements(
                requirements=test_case['requirements'],
                project_path=f"/tmp/benchmark_{i+1}",
                options={
                    'benchmark_mode': True,
                    'production_mode': True,
                    'quality_gate': 'strict'
                }
            )
            
            execution_time = time.time() - start_time
            
            benchmark_result = {
                'name': test_case['name'],
                'success': result.success,
                'execution_time': execution_time,
                'pipeline_time': result.execution_time,
                'quality_score': result.quality_score,
                'files_generated': len(result.generated_files),
                'expected_files': test_case['expected_files']
            }
            
            results.append(benchmark_result)
            
            print(f"  Result: {'✓' if result.success else '✗'}")
            print(f"  Time: {execution_time:.2f}s")
            print(f"  Quality: {result.quality_score:.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'name': test_case['name'],
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            })
    
    # Summary
    print("\n" + "-"*60)
    print("BENCHMARK SUMMARY")
    print("-"*60)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        avg_time = sum(r['execution_time'] for r in successful) / len(successful)
        avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
        
        print(f"Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Average Execution Time: {avg_time:.2f}s")
        print(f"Average Quality Score: {avg_quality:.2f}")
    else:
        print("No successful generations")
    
    return results


async def main():
    """Run all production programming generators."""
    print("CLAUDE-TUI PRODUCTION PROGRAMMING PIPELINE")
    print("=" * 80)
    
    generators = [
        ("Production REST API", generate_production_rest_api),
        ("Production Utility", generate_production_utility),
        ("Production Benchmark", run_production_benchmark)
    ]
    
    for generator_name, generator_func in generators:
        try:
            print(f"\nRunning: {generator_name}")
            await generator_func()
            print(f"\n{generator_name} completed.")
        except Exception as e:
            print(f"\n{generator_name} failed: {e}")
            logger.error(f"Generator {generator_name} failed", exc_info=True)
    
    print("\n" + "="*80)
    print("ALL PRODUCTION GENERATORS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())