import { Reporter, TestResult, TestSuite, TestConfig } from '../types';
import { promises as fs } from 'fs';
import { resolve } from 'path';

export interface JsonReporterOptions {
  outputFile?: string;
  pretty?: boolean;
}

export class JsonReporter implements Reporter {
  private options: JsonReporterOptions;
  private results: TestResult[] = [];
  private config: TestConfig;

  constructor(config: TestConfig, options: JsonReporterOptions = {}) {
    this.config = config;
    this.options = {
      outputFile: 'test-results.json',
      pretty: true,
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
    const report = {
      summary: {
        total: suite.passed + suite.failed + suite.skipped,
        passed: suite.passed,
        failed: suite.failed,
        skipped: suite.skipped,
        duration: suite.duration,
        timestamp: new Date().toISOString()
      },
      coverage: suite.coverage,
      results: this.results.map(result => ({
        file: result.testFilePath,
        success: result.success,
        duration: result.duration,
        error: result.error,
        tests: result.tests.map(test => ({
          name: test.name,
          success: test.success,
          duration: test.duration,
          error: test.error,
          skipped: test.skipped
        })),
        coverage: result.coverage
      }))
    };

    const outputPath = resolve(process.cwd(), this.options.outputFile!);
    const content = this.options.pretty ? 
      JSON.stringify(report, null, 2) : 
      JSON.stringify(report);

    try {
      await fs.writeFile(outputPath, content, 'utf-8');
      if (!this.config.silent) {
        console.log(`📄 JSON report saved to: ${outputPath}`);
      }
    } catch (error) {
      console.error(`❌ Failed to save JSON report: ${error}`);
    }
  }
}