#!/usr/bin/env node

import { Command } from 'commander';
import { TestRunner } from '../runner/TestRunner';
import { ConfigLoader } from '../config/ConfigLoader';
import { FileWatcher } from '../watchers/FileWatcher';
import chalk from 'chalk';
import { TestConfig } from '../types';

const program = new Command();

program
  .name('test-workflow')
  .description('Advanced test framework with parallel execution and watch mode')
  .version('1.0.0');

program
  .option('-c, --config <path>', 'Path to config file', 'test-workflow.config.js')
  .option('-w, --watch', 'Enable watch mode')
  .option('-p, --parallel', 'Run tests in parallel')
  .option('--coverage', 'Collect test coverage')
  .option('--max-workers <number>', 'Maximum number of worker processes', '4')
  .option('--verbose', 'Show verbose output')
  .option('--silent', 'Suppress output except errors')
  .option('--bail', 'Stop after first test failure')
  .option('--reporter <reporters...>', 'Reporters to use', ['console'])
  .option('--timeout <ms>', 'Test timeout in milliseconds', '5000')
  .argument('[patterns...]', 'Test file patterns')
  .action(async (patterns: string[], options) => {
    try {
      console.log(chalk.blue('🚀 Test Workflow Framework\n'));

      const configLoader = new ConfigLoader();
      let config: TestConfig;

      try {
        config = await configLoader.load(options.config);
      } catch (error) {
        console.log(chalk.yellow('⚠️  No config file found, using defaults'));
        config = configLoader.getDefaultConfig();
      }

      // Override config with CLI options
      if (patterns.length > 0) {
        config.testMatch = patterns;
      }
      if (options.parallel) {
        config.maxWorkers = parseInt(options.maxWorkers);
      }
      if (options.coverage) {
        config.collectCoverage = true;
      }
      if (options.verbose) {
        config.verbose = true;
      }
      if (options.silent) {
        config.silent = true;
      }
      if (options.bail) {
        config.bail = true;
      }
      if (options.reporter) {
        config.reporters = options.reporter.map((name: string) => ({ name }));
      }
      if (options.timeout) {
        config.testTimeout = parseInt(options.timeout);
      }
      config.watchMode = options.watch;

      const runner = new TestRunner(config);

      if (options.watch) {
        const watcher = new FileWatcher(config);
        console.log(chalk.cyan('👀 Watch mode enabled. Press Ctrl+C to exit.\n'));
        
        let isRunning = false;
        
        watcher.onChange(async (files: string[]) => {
          if (isRunning) return;
          
          console.log(chalk.yellow(`\n📝 Files changed: ${files.join(', ')}`));
          console.log(chalk.cyan('🔄 Running tests...\n'));
          
          isRunning = true;
          try {
            await runner.run();
          } catch (error) {
            console.error(chalk.red('❌ Test run failed:'), error);
          } finally {
            isRunning = false;
          }
        });

        await watcher.start();
        
        // Run tests initially
        await runner.run();
        
        // Keep process alive
        process.on('SIGINT', async () => {
          console.log(chalk.yellow('\n👋 Stopping watch mode...'));
          await watcher.stop();
          process.exit(0);
        });
        
      } else {
        const result = await runner.run();
        process.exit(result.failed > 0 ? 1 : 0);
      }

    } catch (error) {
      console.error(chalk.red('❌ Fatal error:'), error);
      process.exit(1);
    }
  });

// Add sub-commands
program
  .command('init')
  .description('Initialize test-workflow configuration')
  .action(async () => {
    const configLoader = new ConfigLoader();
    await configLoader.init();
    console.log(chalk.green('✅ Configuration initialized!'));
  });

program
  .command('coverage')
  .description('Generate coverage report only')
  .option('-c, --config <path>', 'Path to config file')
  .action(async (options) => {
    const configLoader = new ConfigLoader();
    const config = await configLoader.load(options.config);
    config.collectCoverage = true;
    
    const runner = new TestRunner(config);
    await runner.run();
  });

if (require.main === module) {
  program.parse();
}

export { program };