import { TestConfig, TestResult, TestSuite, Reporter } from '../types';
import { WorkerPool } from './WorkerPool';
import { ReporterFactory } from '../reporters';
import { CoverageReporter } from '../reporters/CoverageReporter';
import { glob } from 'glob';
import { performance } from 'perf_hooks';
import ora from 'ora';
import chalk from 'chalk';

export class TestRunner {
  private config: TestConfig;
  private reporters: Reporter[] = [];
  private workerPool?: WorkerPool;

  constructor(config: TestConfig) {
    this.config = config;
    this.setupReporters();
  }

  async run(): Promise<TestSuite> {
    const startTime = performance.now();
    const spinner = ora('Discovering test files...').start();

    try {
      // Discover test files
      const testFiles = await this.discoverTestFiles();
      
      if (testFiles.length === 0) {
        spinner.fail('No test files found');
        return this.createEmptySuite();
      }

      spinner.succeed(`Found ${testFiles.length} test file(s)`);

      // Initialize reporters
      for (const reporter of this.reporters) {
        await reporter.onStart?.(this.config);
      }

      let results: TestResult[];

      if (this.config.maxWorkers && this.config.maxWorkers > 1 && testFiles.length > 1) {
        // Run tests in parallel
        results = await this.runTestsInParallel(testFiles);
      } else {
        // Run tests sequentially
        results = await this.runTestsSequentially(testFiles);
      }

      // Calculate suite statistics
      const suite = this.createTestSuite(results, performance.now() - startTime);

      // Notify reporters of completion
      for (const reporter of this.reporters) {
        await reporter.onComplete?.(suite);
      }

      return suite;

    } catch (error) {
      spinner.fail('Test run failed');
      throw error;
    } finally {
      if (this.workerPool) {
        await this.workerPool.shutdown();
      }
    }
  }

  private setupReporters(): void {
    this.reporters = ReporterFactory.createMultiple(this.config, this.config.reporters || []);
  }

  private async discoverTestFiles(): Promise<string[]> {
    const files: string[] = [];
    
    for (const pattern of this.config.testMatch) {
      try {
        const matches = await glob(pattern, {
          ignore: this.config.testIgnore,
          absolute: true
        });
        files.push(...matches);
      } catch (error) {
        if (!this.config.silent) {
          console.warn(`⚠️  Failed to glob pattern ${pattern}:`, error);
        }
      }
    }
    
    return [...new Set(files)]; // Remove duplicates
  }

  private async runTestsInParallel(testFiles: string[]): Promise<TestResult[]> {
    const spinner = ora(`Running ${testFiles.length} test files in parallel (${this.config.maxWorkers} workers)...`).start();
    
    try {
      this.workerPool = new WorkerPool(this.config.maxWorkers!);
      await this.workerPool.initialize();

      const promises = testFiles.map(async (testFile) => {
        try {
          const result = await this.workerPool!.runTest(testFile, this.config);
          
          // Notify reporters of individual test result
          for (const reporter of this.reporters) {
            await reporter.onTestResult?.(result);
          }
          
          if (this.config.bail && !result.success) {
            spinner.fail(`Test failed: ${testFile}`);
            throw new Error(`Test failed and bail is enabled: ${testFile}`);
          }
          
          return result;
        } catch (error) {
          spinner.fail(`Error running test: ${testFile}`);
          return {
            testFilePath: testFile,
            success: false,
            tests: [],
            duration: 0,
            error: error instanceof Error ? error.message : String(error)
          };
        }
      });

      const results = await Promise.all(promises);
      spinner.succeed(`Completed ${testFiles.length} test files`);
      return results;

    } catch (error) {
      spinner.fail('Parallel test execution failed');
      throw error;
    }
  }

  private async runTestsSequentially(testFiles: string[]): Promise<TestResult[]> {
    const results: TestResult[] = [];
    
    const spinner = ora().start();

    for (let i = 0; i < testFiles.length; i++) {
      const testFile = testFiles[i];
      spinner.text = `Running test ${i + 1}/${testFiles.length}: ${testFile}`;
      
      try {
        const result = await this.runSingleTest(testFile);
        results.push(result);
        
        // Notify reporters of individual test result
        for (const reporter of this.reporters) {
          await reporter.onTestResult?.(result);
        }
        
        if (this.config.bail && !result.success) {
          spinner.fail(`Test failed: ${testFile}`);
          break;
        }
        
      } catch (error) {
        const errorResult: TestResult = {
          testFilePath: testFile,
          success: false,
          tests: [],
          duration: 0,
          error: error instanceof Error ? error.message : String(error)
        };
        results.push(errorResult);
        
        if (this.config.bail) {
          spinner.fail(`Test failed: ${testFile}`);
          break;
        }
      }
    }

    spinner.succeed(`Completed ${results.length} test files`);
    return results;
  }

  private async runSingleTest(testFilePath: string): Promise<TestResult> {
    // This is a simplified single test runner
    // In practice, this would import and execute the test file
    
    const startTime = performance.now();
    
    try {
      // Simulate test execution
      await this.executeTestFile(testFilePath);
      
      const duration = performance.now() - startTime;
      
      return {
        testFilePath,
        success: true,
        tests: [
          {
            name: `Test suite in ${testFilePath}`,
            success: true,
            duration: duration
          }
        ],
        duration
      };
    } catch (error) {
      const duration = performance.now() - startTime;
      
      return {
        testFilePath,
        success: false,
        tests: [],
        duration,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  private async executeTestFile(testFilePath: string): Promise<void> {
    // Simplified test file execution
    // This would load and run the actual test file
    // For now, we'll simulate it
    
    const timeout = this.config.testTimeout || 5000;
    
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Test timeout after ${timeout}ms`));
      }, timeout);
      
      try {
        // Simulate async test execution
        setTimeout(() => {
          clearTimeout(timer);
          resolve();
        }, Math.random() * 100 + 50); // Random delay 50-150ms
      } catch (error) {
        clearTimeout(timer);
        reject(error);
      }
    });
  }

  private createTestSuite(results: TestResult[], duration: number): TestSuite {
    const passed = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;
    const skipped = results.reduce((acc, r) => acc + r.tests.filter(t => t.skipped).length, 0);
    
    let coverage;
    if (this.config.collectCoverage) {
      coverage = CoverageReporter.calculateGlobalCoverage(results);
    }

    return {
      name: 'Test Suite',
      results,
      duration,
      passed,
      failed,
      skipped,
      coverage
    };
  }

  private createEmptySuite(): TestSuite {
    return {
      name: 'Test Suite',
      results: [],
      duration: 0,
      passed: 0,
      failed: 0,
      skipped: 0
    };
  }
}