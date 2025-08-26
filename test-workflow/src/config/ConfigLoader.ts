import { TestConfig } from '../types';
import { promises as fs } from 'fs';
import { resolve, dirname } from 'path';
import { pathToFileURL } from 'url';

export class ConfigLoader {
  private defaultConfig: TestConfig = {
    testMatch: ['**/*.test.{js,ts}', '**/*.spec.{js,ts}'],
    testIgnore: ['**/node_modules/**', '**/dist/**', '**/build/**'],
    setupFiles: [],
    setupFilesAfterEnv: [],
    collectCoverage: false,
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
    maxWorkers: 4,
    testTimeout: 5000,
    verbose: false,
    silent: false,
    bail: false,
    reporters: [{ name: 'console' }],
    watchMode: false,
    watchIgnore: ['**/node_modules/**', '**/coverage/**', '**/dist/**']
  };

  public getDefaultConfig(): TestConfig {
    return { ...this.defaultConfig };
  }

  public async load(configPath: string = 'test-workflow.config.js'): Promise<TestConfig> {
    const possiblePaths = [
      configPath,
      'test-workflow.config.js',
      'test-workflow.config.ts',
      'jest.config.js',
      'jest.config.ts',
      'package.json'
    ];

    for (const path of possiblePaths) {
      try {
        const fullPath = resolve(process.cwd(), path);
        await fs.access(fullPath);
        
        if (path === 'package.json') {
          return this.loadFromPackageJson(fullPath);
        }
        
        return this.loadFromConfigFile(fullPath);
      } catch (error) {
        continue;
      }
    }

    throw new Error('No configuration file found');
  }

  private async loadFromConfigFile(filePath: string): Promise<TestConfig> {
    try {
      let config: any;

      if (filePath.endsWith('.ts')) {
        // For TypeScript files, we need to compile and import
        const { register } = await import('ts-node');
        register({
          transpileOnly: true,
          compilerOptions: {
            module: 'commonjs'
          }
        });
      }

      if (filePath.endsWith('.js') || filePath.endsWith('.ts')) {
        delete require.cache[filePath];
        config = require(filePath);
        
        // Handle both default export and module.exports
        if (config.default) {
          config = config.default;
        }
      }

      return this.mergeWithDefaults(config);
    } catch (error) {
      throw new Error(`Failed to load config from ${filePath}: ${error}`);
    }
  }

  private async loadFromPackageJson(filePath: string): Promise<TestConfig> {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const pkg = JSON.parse(content);
      
      const jestConfig = pkg.jest || {};
      const testWorkflowConfig = pkg['test-workflow'] || {};
      
      // Convert Jest config to our format
      const config = {
        ...jestConfig,
        ...testWorkflowConfig
      };

      // Map Jest-specific fields
      if (config.testMatch === undefined && config.testRegex) {
        config.testMatch = Array.isArray(config.testRegex) ? config.testRegex : [config.testRegex];
      }
      
      if (config.testPathIgnorePatterns) {
        config.testIgnore = config.testPathIgnorePatterns;
      }

      return this.mergeWithDefaults(config);
    } catch (error) {
      throw new Error(`Failed to load config from package.json: ${error}`);
    }
  }

  private mergeWithDefaults(userConfig: Partial<TestConfig>): TestConfig {
    const config = { ...this.defaultConfig };

    // Merge arrays by concatenating
    const arrayFields = ['testMatch', 'testIgnore', 'setupFiles', 'setupFilesAfterEnv', 'collectCoverageFrom', 'coverageReporters', 'watchIgnore'];
    
    for (const field of arrayFields) {
      if (userConfig[field as keyof TestConfig]) {
        config[field as keyof TestConfig] = userConfig[field as keyof TestConfig] as any;
      }
    }

    // Merge objects deeply
    if (userConfig.coverageThreshold) {
      config.coverageThreshold = {
        ...config.coverageThreshold,
        ...userConfig.coverageThreshold,
        global: {
          ...config.coverageThreshold!.global,
          ...userConfig.coverageThreshold.global
        }
      };
    }

    // Override simple fields
    const simpleFields = ['collectCoverage', 'coverageDirectory', 'maxWorkers', 'testTimeout', 'verbose', 'silent', 'bail', 'watchMode'];
    
    for (const field of simpleFields) {
      if (userConfig[field as keyof TestConfig] !== undefined) {
        config[field as keyof TestConfig] = userConfig[field as keyof TestConfig] as any;
      }
    }

    // Handle reporters specially
    if (userConfig.reporters) {
      config.reporters = userConfig.reporters.map(reporter => 
        typeof reporter === 'string' ? { name: reporter } : reporter
      );
    }

    return config;
  }

  public async init(): Promise<void> {
    const configContent = `module.exports = {
  // Test file patterns
  testMatch: ['**/*.test.{js,ts}', '**/*.spec.{js,ts}'],
  testIgnore: ['**/node_modules/**', '**/dist/**', '**/build/**'],

  // Setup files
  setupFiles: [],
  setupFilesAfterEnv: [],

  // Coverage configuration
  collectCoverage: false,
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
  silent: false,
  bail: false,

  // Reporters
  reporters: [
    { name: 'console' },
    // { name: 'json', options: { outputFile: 'test-results.json' } },
    // { name: 'html', options: { outputDir: 'test-reports' } }
  ],

  // Watch mode
  watchIgnore: ['**/node_modules/**', '**/coverage/**', '**/dist/**']
};
`;

    await fs.writeFile('test-workflow.config.js', configContent, 'utf-8');
  }
}