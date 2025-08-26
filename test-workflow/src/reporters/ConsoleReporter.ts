import { Reporter, TestResult, TestSuite, TestConfig } from '../types';
import chalk from 'chalk';
import { performance } from 'perf_hooks';

export class ConsoleReporter implements Reporter {
  private startTime: number = 0;
  private config: TestConfig;
  private verbose: boolean = false;

  constructor(config: TestConfig) {
    this.config = config;
    this.verbose = config.verbose || false;
  }

  async onStart(config: TestConfig): Promise<void> {
    this.startTime = performance.now();
    
    if (!config.silent) {
      console.log(chalk.blue('🚀 Starting test run...\n'));
      
      if (this.verbose) {
        console.log(chalk.gray('Configuration:'));
        console.log(chalk.gray(`  Test patterns: ${config.testMatch.join(', ')}`));
        console.log(chalk.gray(`  Max workers: ${config.maxWorkers}`));
        console.log(chalk.gray(`  Coverage: ${config.collectCoverage ? 'enabled' : 'disabled'}`));
        console.log(chalk.gray(`  Watch mode: ${config.watchMode ? 'enabled' : 'disabled'}`));
        console.log('');
      }
    }
  }

  async onTestResult(result: TestResult): Promise<void> {
    if (this.config.silent) return;

    const status = result.success ? chalk.green('✓') : chalk.red('✗');
    const duration = `${result.duration.toFixed(2)}ms`;
    
    if (this.verbose) {
      console.log(`${status} ${result.testFilePath} ${chalk.gray(`(${duration})`)}`);
      
      if (result.tests.length > 0) {
        result.tests.forEach(test => {
          const testStatus = test.success ? chalk.green('  ✓') : test.skipped ? chalk.yellow('  ○') : chalk.red('  ✗');
          const testDuration = `${test.duration.toFixed(2)}ms`;
          console.log(`${testStatus} ${test.name} ${chalk.gray(`(${testDuration})`)}`);
          
          if (test.error && !test.success) {
            console.log(chalk.red(`    ${test.error}`));
          }
        });
        console.log('');
      }
    } else {
      process.stdout.write(result.success ? chalk.green('.') : chalk.red('F'));
    }

    if (result.error && !result.success) {
      console.log(chalk.red(`\n❌ ${result.testFilePath}`));
      console.log(chalk.red(`   ${result.error}\n`));
    }
  }

  async onComplete(suite: TestSuite): Promise<void> {
    const duration = performance.now() - this.startTime;
    
    if (!this.config.silent) {
      if (!this.verbose) {
        console.log('\n');
      }
      
      console.log(chalk.blue('📊 Test Results Summary'));
      console.log(chalk.blue('========================\n'));
      
      const total = suite.passed + suite.failed + suite.skipped;
      console.log(`Tests:       ${this.formatCount(suite.passed, 'passed')} ${this.formatCount(suite.failed, 'failed')} ${this.formatCount(suite.skipped, 'skipped')} ${total} total`);
      console.log(`Time:        ${(duration / 1000).toFixed(2)}s`);
      console.log(`Files:       ${suite.results.length}`);
      
      if (suite.coverage) {
        console.log('\n' + chalk.blue('📈 Coverage Summary'));
        console.log(chalk.blue('=================='));
        this.printCoverageTable(suite.coverage);
      }

      // Print failed tests summary
      const failedResults = suite.results.filter(r => !r.success);
      if (failedResults.length > 0) {
        console.log('\n' + chalk.red('❌ Failed Tests'));
        console.log(chalk.red('==============='));
        
        failedResults.forEach(result => {
          console.log(chalk.red(`\n● ${result.testFilePath}`));
          if (result.error) {
            console.log(chalk.gray(`  ${result.error}`));
          }
          
          const failedTests = result.tests.filter(t => !t.success && !t.skipped);
          failedTests.forEach(test => {
            console.log(chalk.red(`  ✗ ${test.name}`));
            if (test.error) {
              console.log(chalk.gray(`    ${test.error}`));
            }
          });
        });
      }

      // Final status
      console.log('\n' + (suite.failed === 0 ? chalk.green('✅ All tests passed!') : chalk.red(`❌ ${suite.failed} test(s) failed`)));
    }
  }

  private formatCount(count: number, label: string): string {
    if (count === 0) return '';
    
    const color = label === 'passed' ? chalk.green : 
                  label === 'failed' ? chalk.red : 
                  chalk.yellow;
    
    return color(`${count} ${label}`) + ' ';
  }

  private printCoverageTable(coverage: any): void {
    const headers = ['File', 'Stmts', 'Branch', 'Funcs', 'Lines'];
    const rows = [
      ['All files', 
       `${coverage.statements.pct.toFixed(2)}% (${coverage.statements.covered}/${coverage.statements.total})`,
       `${coverage.branches.pct.toFixed(2)}% (${coverage.branches.covered}/${coverage.branches.total})`,
       `${coverage.functions.pct.toFixed(2)}% (${coverage.functions.covered}/${coverage.functions.total})`,
       `${coverage.lines.pct.toFixed(2)}% (${coverage.lines.covered}/${coverage.lines.total})`
      ]
    ];

    const colWidths = headers.map((header, i) => 
      Math.max(header.length, ...rows.map(row => row[i].length))
    );

    // Print header
    const headerRow = headers.map((header, i) => header.padEnd(colWidths[i])).join(' | ');
    console.log(chalk.bold(headerRow));
    console.log(colWidths.map(w => '-'.repeat(w)).join('-|-'));

    // Print rows
    rows.forEach(row => {
      const formattedRow = row.map((cell, i) => cell.padEnd(colWidths[i])).join(' | ');
      console.log(formattedRow);
    });
  }
}