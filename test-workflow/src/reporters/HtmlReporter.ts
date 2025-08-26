import { Reporter, TestResult, TestSuite, TestConfig } from '../types';
import { promises as fs } from 'fs';
import { resolve, join } from 'path';

export interface HtmlReporterOptions {
  outputDir?: string;
  filename?: string;
  title?: string;
}

export class HtmlReporter implements Reporter {
  private options: HtmlReporterOptions;
  private results: TestResult[] = [];
  private config: TestConfig;

  constructor(config: TestConfig, options: HtmlReporterOptions = {}) {
    this.config = config;
    this.options = {
      outputDir: 'test-reports',
      filename: 'index.html',
      title: 'Test Report',
      ...options
    };
  }

  async onStart(config: TestConfig): Promise<void> {
    this.results = [];
  }

  async onTestResult(result: TestResult): Promise<void> {
    this.results.push(result);
  }

  async onComplete(suite: TestSuite): Promise<void> {
    const outputDir = resolve(process.cwd(), this.options.outputDir!);
    const outputPath = join(outputDir, this.options.filename!);

    try {
      await fs.mkdir(outputDir, { recursive: true });
      
      const html = this.generateHtml(suite);
      await fs.writeFile(outputPath, html, 'utf-8');
      
      // Copy CSS file
      await this.copyCssFile(outputDir);
      
      if (!this.config.silent) {
        console.log(`📊 HTML report saved to: ${outputPath}`);
      }
    } catch (error) {
      console.error(`❌ Failed to save HTML report: ${error}`);
    }
  }

  private generateHtml(suite: TestSuite): string {
    const timestamp = new Date().toLocaleString();
    const total = suite.passed + suite.failed + suite.skipped;
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${this.options.title}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>${this.options.title}</h1>
            <p class="timestamp">Generated on ${timestamp}</p>
        </header>

        <section class="summary">
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat ${suite.failed === 0 ? 'success' : 'failure'}">
                    <div class="value">${total}</div>
                    <div class="label">Total</div>
                </div>
                <div class="stat success">
                    <div class="value">${suite.passed}</div>
                    <div class="label">Passed</div>
                </div>
                <div class="stat failure">
                    <div class="value">${suite.failed}</div>
                    <div class="label">Failed</div>
                </div>
                <div class="stat skipped">
                    <div class="value">${suite.skipped}</div>
                    <div class="label">Skipped</div>
                </div>
                <div class="stat">
                    <div class="value">${(suite.duration / 1000).toFixed(2)}s</div>
                    <div class="label">Duration</div>
                </div>
            </div>
        </section>

        ${suite.coverage ? this.generateCoverageSection(suite.coverage) : ''}

        <section class="results">
            <h2>Test Results</h2>
            <div class="test-files">
                ${this.results.map(result => this.generateTestFileSection(result)).join('')}
            </div>
        </section>
    </div>

    <script>
        // Toggle test details
        document.querySelectorAll('.test-file-header').forEach(header => {
            header.addEventListener('click', () => {
                const details = header.nextElementSibling;
                details.style.display = details.style.display === 'none' ? 'block' : 'none';
            });
        });
    </script>
</body>
</html>`;
  }

  private generateCoverageSection(coverage: any): string {
    return `
        <section class="coverage">
            <h2>Coverage</h2>
            <table class="coverage-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Percentage</th>
                        <th>Covered/Total</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Statements</td>
                        <td class="coverage-pct ${this.getCoverageClass(coverage.statements.pct)}">${coverage.statements.pct.toFixed(2)}%</td>
                        <td>${coverage.statements.covered}/${coverage.statements.total}</td>
                    </tr>
                    <tr>
                        <td>Branches</td>
                        <td class="coverage-pct ${this.getCoverageClass(coverage.branches.pct)}">${coverage.branches.pct.toFixed(2)}%</td>
                        <td>${coverage.branches.covered}/${coverage.branches.total}</td>
                    </tr>
                    <tr>
                        <td>Functions</td>
                        <td class="coverage-pct ${this.getCoverageClass(coverage.functions.pct)}">${coverage.functions.pct.toFixed(2)}%</td>
                        <td>${coverage.functions.covered}/${coverage.functions.total}</td>
                    </tr>
                    <tr>
                        <td>Lines</td>
                        <td class="coverage-pct ${this.getCoverageClass(coverage.lines.pct)}">${coverage.lines.pct.toFixed(2)}%</td>
                        <td>${coverage.lines.covered}/${coverage.lines.total}</td>
                    </tr>
                </tbody>
            </table>
        </section>`;
  }

  private generateTestFileSection(result: TestResult): string {
    const statusClass = result.success ? 'success' : 'failure';
    const statusIcon = result.success ? '✓' : '✗';
    
    return `
        <div class="test-file">
            <div class="test-file-header ${statusClass}">
                <span class="status-icon">${statusIcon}</span>
                <span class="file-path">${result.testFilePath}</span>
                <span class="duration">${result.duration.toFixed(2)}ms</span>
            </div>
            <div class="test-file-details" style="display: none;">
                ${result.error ? `<div class="error">${result.error}</div>` : ''}
                <div class="test-cases">
                    ${result.tests.map(test => this.generateTestCaseSection(test)).join('')}
                </div>
            </div>
        </div>`;
  }

  private generateTestCaseSection(test: any): string {
    const statusClass = test.success ? 'success' : test.skipped ? 'skipped' : 'failure';
    const statusIcon = test.success ? '✓' : test.skipped ? '○' : '✗';
    
    return `
        <div class="test-case ${statusClass}">
            <span class="status-icon">${statusIcon}</span>
            <span class="test-name">${test.name}</span>
            <span class="duration">${test.duration.toFixed(2)}ms</span>
            ${test.error ? `<div class="error">${test.error}</div>` : ''}
        </div>`;
  }

  private getCoverageClass(percentage: number): string {
    if (percentage >= 80) return 'high';
    if (percentage >= 60) return 'medium';
    return 'low';
  }

  private async copyCssFile(outputDir: string): Promise<void> {
    const cssContent = `
/* Test Report Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.header h1 {
    margin: 0;
    color: #333;
}

.timestamp {
    color: #666;
    margin: 5px 0 0 0;
}

.summary {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.stats {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.stat {
    text-align: center;
    padding: 15px;
    border-radius: 6px;
    background: #f8f9fa;
    min-width: 80px;
}

.stat.success { background: #d4edda; color: #155724; }
.stat.failure { background: #f8d7da; color: #721c24; }
.stat.skipped { background: #fff3cd; color: #856404; }

.stat .value {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 5px;
}

.stat .label {
    font-size: 14px;
    text-transform: uppercase;
}

.coverage, .results {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.coverage-table {
    width: 100%;
    border-collapse: collapse;
}

.coverage-table th,
.coverage-table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
}

.coverage-pct.high { color: #28a745; font-weight: bold; }
.coverage-pct.medium { color: #ffc107; font-weight: bold; }
.coverage-pct.low { color: #dc3545; font-weight: bold; }

.test-file {
    border: 1px solid #dee2e6;
    border-radius: 6px;
    margin-bottom: 10px;
}

.test-file-header {
    display: flex;
    align-items: center;
    padding: 15px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.test-file-header:hover {
    background-color: #f8f9fa;
}

.test-file-header.success { border-left: 4px solid #28a745; }
.test-file-header.failure { border-left: 4px solid #dc3545; }

.status-icon {
    margin-right: 10px;
    font-weight: bold;
}

.file-path {
    flex: 1;
    font-family: monospace;
}

.duration {
    color: #666;
    font-size: 14px;
}

.test-file-details {
    padding: 0 15px 15px;
    border-top: 1px solid #dee2e6;
}

.error {
    background: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 4px;
    font-family: monospace;
    margin: 10px 0;
    white-space: pre-wrap;
}

.test-case {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f1f1f1;
}

.test-case:last-child {
    border-bottom: none;
}

.test-case.success .status-icon { color: #28a745; }
.test-case.failure .status-icon { color: #dc3545; }
.test-case.skipped .status-icon { color: #ffc107; }

.test-name {
    flex: 1;
    margin-left: 5px;
}
`;

    await fs.writeFile(join(outputDir, 'styles.css'), cssContent, 'utf-8');
  }
}