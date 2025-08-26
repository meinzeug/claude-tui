// Main exports
export { TestRunner } from './runner/TestRunner';
export { WorkerPool } from './runner/WorkerPool';
export { ConfigLoader } from './config/ConfigLoader';
export { FileWatcher, SmartWatcher } from './watchers/FileWatcher';

// Reporters
export {
  ConsoleReporter,
  JsonReporter,
  HtmlReporter,
  CoverageReporter,
  ReporterFactory
} from './reporters';

// Types
export * from './types';

// CLI
export { program } from './cli';

// Default configuration helper
import { ConfigLoader } from './config/ConfigLoader';

export const getDefaultConfig = () => {
  const loader = new ConfigLoader();
  return loader.getDefaultConfig();
};

// Utility function to run tests programmatically
export async function runTests(config?: Partial<import('./types').TestConfig>) {
  const { TestRunner } = await import('./runner/TestRunner');
  const { ConfigLoader } = await import('./config/ConfigLoader');
  
  const configLoader = new ConfigLoader();
  const fullConfig = config ? 
    { ...configLoader.getDefaultConfig(), ...config } :
    configLoader.getDefaultConfig();
  
  const runner = new TestRunner(fullConfig);
  return await runner.run();
}