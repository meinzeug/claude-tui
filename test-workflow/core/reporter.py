"""
Test Reporter System - Comprehensive reporting for test-workflow
Integrates with test runner to provide various output formats
"""

import json
import time
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, TextIO
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import html
import csv
from pathlib import Path


class ReportFormat(Enum):
    CONSOLE = "console"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    JUNIT = "junit"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    format: ReportFormat
    output_file: Optional[str] = None
    include_details: bool = True
    include_stack_traces: bool = True
    include_timing: bool = True
    include_coverage: bool = False
    custom_template: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'format': self.format.value,
            'output_file': self.output_file,
            'include_details': self.include_details,
            'include_stack_traces': self.include_stack_traces,
            'include_timing': self.include_timing,
            'include_coverage': self.include_coverage,
            'custom_template': self.custom_template
        }


class BaseReporter(ABC):
    """Abstract base class for all reporters"""
    
    def __init__(self, config: ReportConfiguration):
        self.config = config
        self.output_buffer: List[str] = []
        
    @abstractmethod
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate report from test results"""
        pass
        
    def write_to_file(self, content: str, file_path: str) -> None:
        """Write content to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"


class ConsoleReporter(BaseReporter):
    """Console output reporter with colors and formatting"""
    
    def __init__(self, config: ReportConfiguration = None):
        if config is None:
            config = ReportConfiguration(ReportFormat.CONSOLE)
        super().__init__(config)
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate colorized console report"""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("TEST EXECUTION REPORT")
        output.append("=" * 80)
        
        # Summary
        summary = test_results.get('summary', {})
        output.append(f"\nSummary:")
        output.append(f"  Total Tests: {summary.get('total', 0)}")
        output.append(f"  Passed:      {self._colorize(summary.get('passed', 0), 'green')} ✓")
        output.append(f"  Failed:      {self._colorize(summary.get('failed', 0), 'red')} ✗")
        output.append(f"  Errors:      {self._colorize(summary.get('errors', 0), 'yellow')} ⚠")
        output.append(f"  Skipped:     {summary.get('skipped', 0)} ⊝")
        output.append(f"  Success Rate: {summary.get('success_rate', 0):.2f}%")
        output.append(f"  Duration:    {self._format_duration(summary.get('duration', 0))}")
        
        # Integration stats
        if 'integration_stats' in test_results:
            stats = test_results['integration_stats']
            output.append(f"\nIntegration Statistics:")
            output.append(f"  Total Assertions: {stats.get('total_assertions', 0)}")
            output.append(f"  Mocks Used:       {stats.get('mocks_used', 0)}")
            output.append(f"  Context Keys:     {stats.get('context_keys', 0)}")
            
        # Test results
        if self.config.include_details:
            output.append(f"\nTest Results:")
            output.append("-" * 80)
            
            for result in test_results.get('results', []):
                status_symbol = self._get_status_symbol(result['status'])
                status_color = self._get_status_color(result['status'])
                
                test_line = f"{status_symbol} {result['name']} [{self._format_duration(result['duration'])}]"
                output.append(self._colorize(test_line, status_color))
                
                if result['message']:
                    output.append(f"    Message: {result['message']}")
                    
                if result['error'] and self.config.include_stack_traces:
                    output.append(f"    Error: {result['error']}")
                    if result['traceback']:
                        lines = result['traceback'].split('\n')
                        for line in lines[-5:]:  # Show last 5 lines of traceback
                            if line.strip():
                                output.append(f"      {line}")
                                
                if result['assertions_count'] > 0:
                    output.append(f"    Assertions: {result['assertions_count']}")
                    
                if result['mocks_used']:
                    output.append(f"    Mocks: {', '.join(result['mocks_used'])}")
                    
                output.append("")  # Empty line between tests
                
        # Footer
        output.append("=" * 80)
        output.append(f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        report_content = "\n".join(output)
        
        if self.config.output_file:
            self.write_to_file(report_content, self.config.output_file)
            
        return report_content
        
    def _colorize(self, text: Union[str, int], color: str) -> str:
        """Add ANSI color codes to text"""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        
        color_code = colors.get(color, colors['white'])
        reset_code = colors['reset']
        return f"{color_code}{text}{reset_code}"
        
    def _get_status_symbol(self, status: str) -> str:
        """Get symbol for test status"""
        symbols = {
            'passed': '✓',
            'failed': '✗',
            'error': '⚠',
            'skipped': '⊝',
            'pending': '○',
            'running': '⟳'
        }
        return symbols.get(status, '?')
        
    def _get_status_color(self, status: str) -> str:
        """Get color for test status"""
        colors = {
            'passed': 'green',
            'failed': 'red',
            'error': 'yellow',
            'skipped': 'cyan',
            'pending': 'blue',
            'running': 'purple'
        }
        return colors.get(status, 'white')


class JsonReporter(BaseReporter):
    """JSON format reporter for machine consumption"""
    
    def __init__(self, config: ReportConfiguration = None):
        if config is None:
            config = ReportConfiguration(ReportFormat.JSON)
        super().__init__(config)
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate JSON report"""
        report_data = {
            'test_framework': 'test-workflow',
            'version': '1.0.0',
            'generated_at': time.time(),
            'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': test_results
        }
        
        report_content = json.dumps(report_data, indent=2, default=str)
        
        if self.config.output_file:
            self.write_to_file(report_content, self.config.output_file)
            
        return report_content


class JUnitReporter(BaseReporter):
    """JUnit XML format reporter for CI/CD integration"""
    
    def __init__(self, config: ReportConfiguration = None):
        if config is None:
            config = ReportConfiguration(ReportFormat.JUNIT)
        super().__init__(config)
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate JUnit XML report"""
        root = ET.Element('testsuites')
        
        # Add summary attributes
        summary = test_results.get('summary', {})
        root.set('tests', str(summary.get('total', 0)))
        root.set('failures', str(summary.get('failed', 0)))
        root.set('errors', str(summary.get('errors', 0)))
        root.set('time', str(summary.get('duration', 0)))
        root.set('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S'))
        
        # Group tests by suite (assuming suite name is in test name)
        suites = {}
        for result in test_results.get('results', []):
            if '.' in result['name']:
                suite_name, test_name = result['name'].split('.', 1)
            else:
                suite_name, test_name = 'default', result['name']
                
            if suite_name not in suites:
                suites[suite_name] = []
            suites[suite_name].append((test_name, result))
            
        # Create testsuite elements
        for suite_name, tests in suites.items():
            testsuite = ET.SubElement(root, 'testsuite')
            testsuite.set('name', suite_name)
            testsuite.set('tests', str(len(tests)))
            
            suite_failures = sum(1 for _, result in tests if result['status'] == 'failed')
            suite_errors = sum(1 for _, result in tests if result['status'] == 'error')
            suite_time = sum(result['duration'] for _, result in tests)
            
            testsuite.set('failures', str(suite_failures))
            testsuite.set('errors', str(suite_errors))
            testsuite.set('time', str(suite_time))
            
            # Add test cases
            for test_name, result in tests:
                testcase = ET.SubElement(testsuite, 'testcase')
                testcase.set('name', test_name)
                testcase.set('classname', suite_name)
                testcase.set('time', str(result['duration']))
                
                if result['status'] == 'failed':
                    failure = ET.SubElement(testcase, 'failure')
                    failure.set('message', result['message'])
                    failure.text = result['traceback']
                elif result['status'] == 'error':
                    error = ET.SubElement(testcase, 'error')
                    error.set('message', result['message'])
                    error.text = result['traceback']
                elif result['status'] == 'skipped':
                    skipped = ET.SubElement(testcase, 'skipped')
                    skipped.set('message', result['message'])
                    
        # Convert to string
        report_content = ET.tostring(root, encoding='unicode', xml_declaration=True)
        
        # Pretty print
        import xml.dom.minidom
        dom = xml.dom.minidom.parseString(report_content)
        report_content = dom.toprettyxml(indent="  ")
        
        if self.config.output_file:
            self.write_to_file(report_content, self.config.output_file)
            
        return report_content


class HtmlReporter(BaseReporter):
    """HTML format reporter with interactive features"""
    
    def __init__(self, config: ReportConfiguration = None):
        if config is None:
            config = ReportConfiguration(ReportFormat.HTML)
        super().__init__(config)
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        summary = test_results.get('summary', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .stat {{
            text-align: center;
            padding: 15px;
            border-radius: 6px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .error {{ color: #ffc107; }}
        .skipped {{ color: #6c757d; }}
        .test-results {{
            padding: 20px;
        }}
        .test-item {{
            border: 1px solid #e9ecef;
            border-radius: 6px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        .test-header {{
            padding: 15px;
            background-color: #f8f9fa;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .test-header:hover {{
            background-color: #e9ecef;
        }}
        .test-details {{
            padding: 15px;
            border-top: 1px solid #e9ecef;
            display: none;
        }}
        .status-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status-error {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .status-skipped {{
            background-color: #e2e3e5;
            color: #383d41;
        }}
        .traceback {{
            background-color: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Results Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{summary.get('total', 0)}</div>
                <div>Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value passed">{summary.get('passed', 0)}</div>
                <div>Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{summary.get('failed', 0)}</div>
                <div>Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value error">{summary.get('errors', 0)}</div>
                <div>Errors</div>
            </div>
            <div class="stat">
                <div class="stat-value skipped">{summary.get('skipped', 0)}</div>
                <div>Skipped</div>
            </div>
            <div class="stat">
                <div class="stat-value">{summary.get('success_rate', 0):.1f}%</div>
                <div>Success Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self._format_duration(summary.get('duration', 0))}</div>
                <div>Total Time</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
"""
        
        # Add test results
        for result in test_results.get('results', []):
            status_class = f"status-{result['status']}"
            html_content += f"""
            <div class="test-item">
                <div class="test-header" onclick="toggleDetails(this)">
                    <span>
                        <strong>{html.escape(result['name'])}</strong>
                        <small>({self._format_duration(result['duration'])})</small>
                    </span>
                    <span class="status-badge {status_class}">{result['status'].upper()}</span>
                </div>
                <div class="test-details">
                    {f'<p><strong>Message:</strong> {html.escape(result["message"])}</p>' if result['message'] else ''}
                    {f'<p><strong>Assertions:</strong> {result["assertions_count"]}</p>' if result['assertions_count'] > 0 else ''}
                    {f'<p><strong>Mocks Used:</strong> {", ".join(result["mocks_used"])}</p>' if result['mocks_used'] else ''}
                    {f'<div class="traceback">{html.escape(result["traceback"])}</div>' if result['traceback'] and self.config.include_stack_traces else ''}
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="footer">
            Generated by test-workflow framework
        </div>
    </div>
    
    <script>
        function toggleDetails(header) {
            const details = header.nextElementSibling;
            if (details.style.display === 'none' || details.style.display === '') {
                details.style.display = 'block';
            } else {
                details.style.display = 'none';
            }
        }
        
        // Expand failed and error tests by default
        document.addEventListener('DOMContentLoaded', function() {
            const headers = document.querySelectorAll('.test-header');
            headers.forEach(header => {
                const badge = header.querySelector('.status-badge');
                if (badge && (badge.textContent === 'FAILED' || badge.textContent === 'ERROR')) {
                    toggleDetails(header);
                }
            });
        });
    </script>
</body>
</html>
"""
        
        if self.config.output_file:
            self.write_to_file(html_content, self.config.output_file)
            
        return html_content


class MarkdownReporter(BaseReporter):
    """Markdown format reporter for documentation integration"""
    
    def __init__(self, config: ReportConfiguration = None):
        if config is None:
            config = ReportConfiguration(ReportFormat.MARKDOWN)
        super().__init__(config)
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        output = []
        
        # Header
        output.append("# Test Results Report")
        output.append(f"*Generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}*")
        output.append("")
        
        # Summary
        summary = test_results.get('summary', {})
        output.append("## Summary")
        output.append("")
        output.append("| Metric | Value |")
        output.append("|--------|-------|")
        output.append(f"| Total Tests | {summary.get('total', 0)} |")
        output.append(f"| Passed | {summary.get('passed', 0)} ✅ |")
        output.append(f"| Failed | {summary.get('failed', 0)} ❌ |")
        output.append(f"| Errors | {summary.get('errors', 0)} ⚠️ |")
        output.append(f"| Skipped | {summary.get('skipped', 0)} ⏭️ |")
        output.append(f"| Success Rate | {summary.get('success_rate', 0):.2f}% |")
        output.append(f"| Total Duration | {self._format_duration(summary.get('duration', 0))} |")
        output.append("")
        
        # Integration stats
        if 'integration_stats' in test_results:
            stats = test_results['integration_stats']
            output.append("## Integration Statistics")
            output.append("")
            output.append("| Metric | Value |")
            output.append("|--------|-------|")
            output.append(f"| Total Assertions | {stats.get('total_assertions', 0)} |")
            output.append(f"| Mocks Used | {stats.get('mocks_used', 0)} |")
            output.append(f"| Context Keys | {stats.get('context_keys', 0)} |")
            output.append("")
            
        # Test details
        if self.config.include_details:
            output.append("## Test Results")
            output.append("")
            
            for result in test_results.get('results', []):
                status_emoji = {
                    'passed': '✅',
                    'failed': '❌', 
                    'error': '⚠️',
                    'skipped': '⏭️'
                }.get(result['status'], '❓')
                
                output.append(f"### {status_emoji} {result['name']}")
                output.append("")
                output.append(f"- **Status**: {result['status'].title()}")
                output.append(f"- **Duration**: {self._format_duration(result['duration'])}")
                
                if result['message']:
                    output.append(f"- **Message**: {result['message']}")
                    
                if result['assertions_count'] > 0:
                    output.append(f"- **Assertions**: {result['assertions_count']}")
                    
                if result['mocks_used']:
                    output.append(f"- **Mocks**: {', '.join(result['mocks_used'])}")
                    
                if result['error'] and self.config.include_stack_traces:
                    output.append("")
                    output.append("**Error Details:**")
                    output.append("```")
                    output.append(result['traceback'] or str(result['error']))
                    output.append("```")
                    
                output.append("")
                
        # Footer
        output.append("---")
        output.append("*Report generated by test-workflow framework*")
        
        report_content = "\n".join(output)
        
        if self.config.output_file:
            self.write_to_file(report_content, self.config.output_file)
            
        return report_content


class CompositeReporter(BaseReporter):
    """Composite reporter that delegates to multiple reporters"""
    
    def __init__(self, reporters: List[BaseReporter]):
        self.reporters = reporters
        super().__init__(ReportConfiguration(ReportFormat.CONSOLE))  # Dummy config
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate reports using all configured reporters"""
        results = []
        
        for reporter in self.reporters:
            try:
                result = await reporter.generate_report(test_results)
                results.append(result)
            except Exception as e:
                print(f"Reporter {reporter.__class__.__name__} failed: {e}")
                
        return "\n\n".join(results)


class TestReporter:
    """Main reporter orchestrator"""
    
    def __init__(self):
        self.reporters: List[BaseReporter] = []
        
    def add_reporter(self, reporter: BaseReporter) -> None:
        """Add a reporter"""
        self.reporters.append(reporter)
        
    def configure_reporter(
        self,
        format_type: ReportFormat,
        output_file: str = None,
        **kwargs
    ) -> BaseReporter:
        """Configure and add a reporter"""
        config = ReportConfiguration(
            format=format_type,
            output_file=output_file,
            **kwargs
        )
        
        reporter_classes = {
            ReportFormat.CONSOLE: ConsoleReporter,
            ReportFormat.JSON: JsonReporter,
            ReportFormat.JUNIT: JUnitReporter,
            ReportFormat.HTML: HtmlReporter,
            ReportFormat.MARKDOWN: MarkdownReporter
        }
        
        reporter_class = reporter_classes.get(format_type)
        if not reporter_class:
            raise ValueError(f"Unsupported report format: {format_type}")
            
        reporter = reporter_class(config)
        self.add_reporter(reporter)
        return reporter
        
    async def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate reports using all configured reporters"""
        if not self.reporters:
            # Default to console reporter
            self.configure_reporter(ReportFormat.CONSOLE)
            
        composite = CompositeReporter(self.reporters)
        return await composite.generate_report(test_results)