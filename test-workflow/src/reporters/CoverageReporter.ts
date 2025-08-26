import { Reporter, TestResult, TestSuite, TestConfig, CoverageData } from '../types';
import { createCoverageMap, CoverageMap } from 'istanbul-lib-coverage';
import { createContext } from 'istanbul-lib-report';
import * as reports from 'istanbul-reports';
import { promises as fs } from 'fs';
import { resolve } from 'path';

export interface CoverageReporterOptions {
  reporters?: string[];
  directory?: string;
  includeAllSources?: boolean;
}

export class CoverageReporter implements Reporter {
  private options: CoverageReporterOptions;
  private config: TestConfig;
  private coverageMap: CoverageMap;

  constructor(config: TestConfig, options: CoverageReporterOptions = {}) {
    this.config = config;
    this.options = {
      reporters: ['text', 'lcov', 'html'],
      directory: 'coverage',
      includeAllSources: false,
      ...options
    };
    this.coverageMap = createCoverageMap({});
  }

  async onStart(config: TestConfig): Promise<void> {
    this.coverageMap = createCoverageMap({});
  }

  async onTestResult(result: TestResult): Promise<void> {
    if (result.coverage && this.config.collectCoverage) {
      // Merge coverage data into the global coverage map
      this.mergeCoverage(result.coverage, result.testFilePath);
    }
  }

  async onComplete(suite: TestSuite): Promise<void> {
    if (!this.config.collectCoverage) return;

    try {
      const outputDir = resolve(process.cwd(), this.options.directory!);
      await fs.mkdir(outputDir, { recursive: true });

      const context = createContext({
        dir: outputDir,
        coverageMap: this.coverageMap,
        sourceFinder: this.createSourceFinder()
      });

      // Generate reports
      for (const reporterName of this.options.reporters!) {
        try {
          const reporter = reports.create(reporterName as any, {});
          reporter.execute(context);
          
          if (!this.config.silent) {
            console.log(`📊 Coverage report (${reporterName}) generated in: ${outputDir}`);
          }
        } catch (error) {
          console.error(`❌ Failed to generate ${reporterName} coverage report:`, error);
        }
      }

      // Check coverage thresholds
      if (this.config.coverageThreshold?.global) {
        await this.checkThresholds(suite.coverage);
      }

    } catch (error) {
      console.error('❌ Failed to generate coverage reports:', error);
    }
  }

  private mergeCoverage(coverage: CoverageData, testFilePath: string): void {
    // Convert our coverage data to Istanbul format
    // This is a simplified implementation - in practice, you'd need
    // to collect actual coverage data from instrumented code
    const fileCoverage = {
      path: testFilePath,
      statementMap: {},
      fnMap: {},
      branchMap: {},
      s: {},
      f: {},
      b: {}
    };

    this.coverageMap.addFileCoverage(fileCoverage);
  }

  private createSourceFinder() {
    return (filePath: string) => {
      try {
        return require('fs').readFileSync(filePath, 'utf8');
      } catch (error) {
        return '';
      }
    };
  }

  private async checkThresholds(coverage?: CoverageData): Promise<void> {
    if (!coverage || !this.config.coverageThreshold?.global) return;

    const thresholds = this.config.coverageThreshold.global;
    const failures: string[] = [];

    if (thresholds.statements && coverage.statements.pct < thresholds.statements) {
      failures.push(`Statements coverage (${coverage.statements.pct.toFixed(2)}%) below threshold (${thresholds.statements}%)`);
    }

    if (thresholds.branches && coverage.branches.pct < thresholds.branches) {
      failures.push(`Branches coverage (${coverage.branches.pct.toFixed(2)}%) below threshold (${thresholds.branches}%)`);
    }

    if (thresholds.functions && coverage.functions.pct < thresholds.functions) {
      failures.push(`Functions coverage (${coverage.functions.pct.toFixed(2)}%) below threshold (${thresholds.functions}%)`);
    }

    if (thresholds.lines && coverage.lines.pct < thresholds.lines) {
      failures.push(`Lines coverage (${coverage.lines.pct.toFixed(2)}%) below threshold (${thresholds.lines}%)`);
    }

    if (failures.length > 0) {
      console.log('\n❌ Coverage threshold failures:');
      failures.forEach(failure => console.log(`  ${failure}`));
      
      if (this.config.bail) {
        process.exit(1);
      }
    } else {
      console.log('\n✅ All coverage thresholds met');
    }
  }

  public static calculateGlobalCoverage(results: TestResult[]): CoverageData {
    const totals = {
      statements: { total: 0, covered: 0 },
      branches: { total: 0, covered: 0 },
      functions: { total: 0, covered: 0 },
      lines: { total: 0, covered: 0 }
    };

    results.forEach(result => {
      if (result.coverage) {
        totals.statements.total += result.coverage.statements.total;
        totals.statements.covered += result.coverage.statements.covered;
        totals.branches.total += result.coverage.branches.total;
        totals.branches.covered += result.coverage.branches.covered;
        totals.functions.total += result.coverage.functions.total;
        totals.functions.covered += result.coverage.functions.covered;
        totals.lines.total += result.coverage.lines.total;
        totals.lines.covered += result.coverage.lines.covered;
      }
    });

    return {
      statements: {
        ...totals.statements,
        skipped: 0,
        pct: totals.statements.total > 0 ? (totals.statements.covered / totals.statements.total) * 100 : 0
      },
      branches: {
        ...totals.branches,
        skipped: 0,
        pct: totals.branches.total > 0 ? (totals.branches.covered / totals.branches.total) * 100 : 0
      },
      functions: {
        ...totals.functions,
        skipped: 0,
        pct: totals.functions.total > 0 ? (totals.functions.covered / totals.functions.total) * 100 : 0
      },
      lines: {
        ...totals.lines,
        skipped: 0,
        pct: totals.lines.total > 0 ? (totals.lines.covered / totals.lines.total) * 100 : 0
      }
    };
  }
}