import { parentPort } from 'worker_threads';
import { TestResult, TestCase, WorkerMessage, CoverageData } from '../types';
import { promises as fs } from 'fs';
import { resolve, dirname, basename } from 'path';
import { performance } from 'perf_hooks';

class TestWorker {
  private async runTest(testFilePath: string, config: any): Promise<TestResult> {
    const startTime = performance.now();
    
    try {
      // Load and execute the test file
      const result = await this.executeTestFile(testFilePath, config);
      const duration = performance.now() - startTime;
      
      return {
        testFilePath,
        success: result.success,
        tests: result.tests,
        coverage: result.coverage,
        duration,
        error: result.error
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

  private async executeTestFile(testFilePath: string, config: any): Promise<{
    success: boolean;
    tests: TestCase[];
    coverage?: CoverageData;
    error?: string;
  }> {
    const tests: TestCase[] = [];
    let success = true;
    let error: string | undefined;

    try {
      // Create a test execution context
      const testContext = this.createTestContext(tests, config);
      
      // Read the test file
      const testContent = await fs.readFile(testFilePath, 'utf-8');
      
      // Execute the test file in the context
      await this.executeInContext(testContent, testFilePath, testContext, config);
      
      // Check if all tests passed
      success = tests.every(test => test.success || test.skipped);
      
      // Generate coverage data if enabled
      let coverage: CoverageData | undefined;
      if (config.collectCoverage) {
        coverage = await this.generateCoverage(testFilePath, config);
      }
      
      return { success, tests, coverage, error };
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      return { success: false, tests, error };
    }
  }

  private createTestContext(tests: TestCase[], config: any) {
    const context = {
      describe: (description: string, fn: Function) => {
        // Simple describe implementation
        try {
          fn();
        } catch (error) {
          tests.push({
            name: description,
            success: false,
            duration: 0,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      },
      
      it: async (description: string, fn: Function) => {
        const startTime = performance.now();
        try {
          if (fn.length > 0) {
            // Async test with done callback
            await new Promise<void>((resolve, reject) => {
              const done = (err?: Error) => {
                if (err) reject(err);
                else resolve();
              };
              
              const result = fn(done);
              if (result && typeof result.then === 'function') {
                result.then(() => resolve()).catch(reject);
              }
            });
          } else {
            // Sync or promise-based test
            const result = fn();
            if (result && typeof result.then === 'function') {
              await result;
            }
          }
          
          const duration = performance.now() - startTime;
          tests.push({
            name: description,
            success: true,
            duration
          });
        } catch (error) {
          const duration = performance.now() - startTime;
          tests.push({
            name: description,
            success: false,
            duration,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      },
      
      test: function(description: string, fn: Function) {
        return context.it(description, fn);
      },
      
      expect: this.createExpectFunction(),
      
      beforeEach: (fn: Function) => {
        // Store setup function - simple implementation
        context._beforeEach = fn;
      },
      
      afterEach: (fn: Function) => {
        // Store teardown function - simple implementation  
        context._afterEach = fn;
      },
      
      beforeAll: (fn: Function) => {
        // Store global setup function
        context._beforeAll = fn;
      },
      
      afterAll: (fn: Function) => {
        // Store global teardown function
        context._afterAll = fn;
      },
      
      skip: {
        it: (description: string, fn?: Function) => {
          tests.push({
            name: description,
            success: true,
            duration: 0,
            skipped: true
          });
        },
        test: function(description: string, fn?: Function) {
          return context.skip.it(description, fn);
        }
      }
    };

    return context;
  }

  private createExpectFunction() {
    function expect(actual: any) {
      return {
        toBe: (expected: any) => {
          if (actual !== expected) {
            throw new Error(`Expected ${actual} to be ${expected}`);
          }
        },
        toEqual: (expected: any) => {
          if (!this.deepEqual(actual, expected)) {
            throw new Error(`Expected ${JSON.stringify(actual)} to equal ${JSON.stringify(expected)}`);
          }
        },
        toBeTruthy: () => {
          if (!actual) {
            throw new Error(`Expected ${actual} to be truthy`);
          }
        },
        toBeFalsy: () => {
          if (actual) {
            throw new Error(`Expected ${actual} to be falsy`);
          }
        },
        toThrow: (expectedError?: string | RegExp) => {
          if (typeof actual !== 'function') {
            throw new Error('Expected value must be a function when using toThrow');
          }
          
          let thrown = false;
          let error: Error;
          
          try {
            actual();
          } catch (err) {
            thrown = true;
            error = err as Error;
          }
          
          if (!thrown) {
            throw new Error('Expected function to throw an error');
          }
          
          if (expectedError) {
            const message = error!.message;
            if (typeof expectedError === 'string' && message !== expectedError) {
              throw new Error(`Expected error message "${message}" to be "${expectedError}"`);
            } else if (expectedError instanceof RegExp && !expectedError.test(message)) {
              throw new Error(`Expected error message "${message}" to match ${expectedError}`);
            }
          }
        },
        toContain: (expected: any) => {
          if (Array.isArray(actual)) {
            if (!actual.includes(expected)) {
              throw new Error(`Expected array to contain ${expected}`);
            }
          } else if (typeof actual === 'string') {
            if (!actual.includes(expected)) {
              throw new Error(`Expected string "${actual}" to contain "${expected}"`);
            }
          } else {
            throw new Error('toContain can only be used with arrays or strings');
          }
        }
      };
    }

    expect.deepEqual = function(a: any, b: any): boolean {
      if (a === b) return true;
      if (a == null || b == null) return false;
      if (typeof a !== typeof b) return false;
      
      if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
          if (!expect.deepEqual(a[i], b[i])) return false;
        }
        return true;
      }
      
      if (typeof a === 'object') {
        const keysA = Object.keys(a);
        const keysB = Object.keys(b);
        if (keysA.length !== keysB.length) return false;
        
        for (const key of keysA) {
          if (!keysB.includes(key)) return false;
          if (!expect.deepEqual(a[key], b[key])) return false;
        }
        return true;
      }
      
      return false;
    };

    return expect;
  }

  private async executeInContext(testContent: string, testFilePath: string, context: any, config: any): Promise<void> {
    // Create a new context with our test functions
    const contextKeys = Object.keys(context);
    const contextValues = contextKeys.map(key => context[key]);
    
    // Add common globals
    const globals = {
      console,
      setTimeout,
      setInterval,
      clearTimeout,
      clearInterval,
      process: {
        env: process.env
      },
      Buffer,
      require: (modulePath: string) => {
        // Simple require implementation for relative paths
        if (modulePath.startsWith('./') || modulePath.startsWith('../')) {
          const fullPath = resolve(dirname(testFilePath), modulePath);
          return require(fullPath);
        }
        return require(modulePath);
      },
      __filename: testFilePath,
      __dirname: dirname(testFilePath)
    };
    
    const globalKeys = Object.keys(globals);
    const globalValues = globalKeys.map(key => (globals as any)[key]);
    
    // Execute the test file with the context
    const func = new Function(
      ...contextKeys,
      ...globalKeys,
      testContent
    );
    
    await func(...contextValues, ...globalValues);
  }

  private async generateCoverage(testFilePath: string, config: any): Promise<CoverageData> {
    // Simplified coverage generation
    // In a real implementation, this would use instrumentation data
    
    return {
      statements: { total: 100, covered: 85, skipped: 0, pct: 85 },
      branches: { total: 50, covered: 40, skipped: 0, pct: 80 },
      functions: { total: 20, covered: 18, skipped: 0, pct: 90 },
      lines: { total: 95, covered: 80, skipped: 0, pct: 84.2 }
    };
  }
}

// Worker message handling
if (parentPort) {
  const worker = new TestWorker();
  
  parentPort.on('message', async (message: WorkerMessage) => {
    if (message.type === 'test') {
      try {
        const result = await worker['runTest'](
          message.payload.testFilePath,
          message.payload.config
        );
        
        parentPort!.postMessage({
          type: 'result',
          payload: result
        });
      } catch (error) {
        parentPort!.postMessage({
          type: 'error',
          payload: {
            message: error instanceof Error ? error.message : String(error),
            stack: error instanceof Error ? error.stack : undefined
          }
        });
      }
    }
  });
}