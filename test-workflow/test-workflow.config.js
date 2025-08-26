module.exports = {
  // Test file patterns
  testMatch: [
    'examples/**/*.test.{js,ts}',
    'src/**/*.test.{js,ts}',
    'tests/**/*.{js,ts}'
  ],
  testIgnore: ['**/node_modules/**', '**/dist/**', '**/build/**'],

  // Setup files
  setupFiles: [],
  setupFilesAfterEnv: [],

  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: [
    'examples/**/*.{js,ts}',
    '!examples/**/*.test.{js,ts}',
    'src/**/*.{js,ts}',
    '!src/**/*.test.{js,ts}'
  ],
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
  testTimeout: 10000,
  verbose: true,
  silent: false,
  bail: false,

  // Reporters
  reporters: [
    { name: 'console' },
    { name: 'json', options: { outputFile: 'test-results.json' } },
    { name: 'html', options: { outputDir: 'test-reports' } }
  ],

  // Watch mode
  watchIgnore: ['**/node_modules/**', '**/coverage/**', '**/dist/**', '**/test-reports/**']
};