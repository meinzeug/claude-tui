# Test Workflow Framework

A comprehensive test framework with parallel execution, watch mode, multiple reporters, and coverage analysis.

## Features

- 🚀 **Parallel Test Execution** - Run tests concurrently using worker threads
- 👀 **Watch Mode** - Automatically re-run tests when files change
- 📊 **Multiple Reporters** - Console, JSON, HTML, and Coverage reports
- ⚙️ **Jest-Compatible Config** - Supports jest.config.js and package.json configuration
- 🎯 **TypeScript Support** - Full TypeScript support with type definitions
- 🔧 **Flexible Configuration** - Extensive configuration options
- 📈 **Coverage Analysis** - Built-in code coverage with threshold checking

## Installation

```bash
npm install test-workflow
# or
yarn add test-workflow
```

## Quick Start

### 1. Initialize Configuration

```bash
npx test-workflow init
```

This creates a `test-workflow.config.js` file with default settings.

### 2. Write Your First Test

```javascript
// math.test.js
describe('Math Operations', () => {
  it('should add numbers correctly', () => {
    expect(2 + 3).toBe(5);
  });
  
  it('should handle async operations', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });
});
```

### 3. Run Tests

```bash
# Run all tests
npx test-workflow

# Run with coverage
npx test-workflow --coverage

# Run in watch mode
npx test-workflow --watch

# Run with parallel execution
npx test-workflow --parallel --max-workers 8
```

## Command Line Interface

```bash
test-workflow [options] [patterns...]

Options:
  -c, --config <path>        Path to config file (default: test-workflow.config.js)
  -w, --watch               Enable watch mode
  -p, --parallel            Run tests in parallel
  --coverage                Collect test coverage
  --max-workers <number>    Maximum number of worker processes (default: 4)
  --verbose                 Show verbose output
  --silent                  Suppress output except errors
  --bail                    Stop after first test failure
  --reporter <reporters...> Reporters to use (default: console)
  --timeout <ms>            Test timeout in milliseconds (default: 5000)
  -h, --help                Display help information
  -V, --version             Display version number

Commands:
  init                      Initialize test-workflow configuration
  coverage                  Generate coverage report only
```

## Configuration

### test-workflow.config.js

```javascript
module.exports = {
  // Test file patterns
  testMatch: ['**/*.test.{js,ts}', '**/*.spec.{js,ts}'],
  testIgnore: ['**/node_modules/**', '**/dist/**'],

  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: ['src/**/*.{js,ts}'],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },

  // Execution configuration
  maxWorkers: 4,
  testTimeout: 5000,
  verbose: false,
  bail: false,

  // Reporters
  reporters: [
    { name: 'console' },
    { name: 'json', options: { outputFile: 'test-results.json' } },
    { name: 'html', options: { outputDir: 'test-reports' } }
  ],

  // Watch mode
  watchIgnore: ['**/node_modules/**', '**/coverage/**']
};
```

### Jest Compatibility

The framework supports Jest-style configuration in `package.json`:

```json
{
  "jest": {
    "testMatch": ["**/*.test.js"],
    "collectCoverage": true,
    "coverageReporters": ["text", "html"]
  }
}
```

## Test API

### Basic Assertions

```javascript
describe('Test Suite', () => {
  it('should pass', () => {
    expect(true).toBeTruthy();
    expect(false).toBeFalsy();
    expect(5).toBe(5);
    expect({ a: 1 }).toEqual({ a: 1 });
    expect([1, 2, 3]).toContain(2);
    expect(() => { throw new Error('test'); }).toThrow('test');
  });
});
```

### Async Testing

```javascript
describe('Async Tests', () => {
  it('should handle promises', async () => {
    const result = await Promise.resolve('success');
    expect(result).toBe('success');
  });
  
  it('should handle callbacks', (done) => {
    setTimeout(() => {
      expect(true).toBeTruthy();
      done();
    }, 100);
  });
});
```

### Setup and Teardown

```javascript
describe('Setup/Teardown', () => {
  beforeAll(async () => {
    // Global setup
  });
  
  afterAll(async () => {
    // Global cleanup
  });
  
  beforeEach(() => {
    // Setup before each test
  });
  
  afterEach(() => {
    // Cleanup after each test
  });
  
  it('should run test with setup', () => {
    // Test implementation
  });
});
```

## Reporters

### Console Reporter

Provides colorful console output with test results and coverage summary.

### JSON Reporter

Generates detailed JSON reports suitable for CI/CD integration:

```javascript
{
  name: 'json',
  options: {
    outputFile: 'test-results.json',
    pretty: true
  }
}
```

### HTML Reporter

Creates comprehensive HTML reports with interactive features:

```javascript
{
  name: 'html',
  options: {
    outputDir: 'test-reports',
    title: 'Test Results'
  }
}
```

### Coverage Reporter

Generates coverage reports in multiple formats (text, lcov, html):

```javascript
{
  name: 'coverage',
  options: {
    reporters: ['text', 'lcov', 'html'],
    directory: 'coverage'
  }
}
```

## Programmatic Usage

```javascript
import { runTests, TestRunner, ConfigLoader } from 'test-workflow';

// Simple usage
const results = await runTests({
  testMatch: ['**/*.test.js'],
  collectCoverage: true
});

// Advanced usage
const configLoader = new ConfigLoader();
const config = await configLoader.load('my-config.js');
const runner = new TestRunner(config);
const suite = await runner.run();
```

## Watch Mode

Watch mode automatically re-runs tests when files change:

```bash
npx test-workflow --watch
```

Features:
- Intelligent file watching with dependency tracking
- Debounced execution to avoid excessive runs
- Configurable ignore patterns
- Support for both test files and source files

## Parallel Execution

Run tests in parallel for faster execution:

```bash
npx test-workflow --parallel --max-workers 8
```

Features:
- Worker thread-based parallel execution
- Automatic worker management
- Load balancing across workers
- Fault tolerance and recovery

## Coverage Analysis

Generate comprehensive code coverage reports:

```bash
npx test-workflow --coverage
```

Features:
- Line, branch, function, and statement coverage
- Multiple output formats (text, lcov, html)
- Configurable thresholds
- Integration with popular coverage tools

## Examples

Check the `examples/` directory for comprehensive examples:

- `basic/` - Simple test cases
- `async/` - Promise and callback testing
- `advanced/` - Setup/teardown and complex scenarios
- `coverage/` - Coverage demonstration

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details