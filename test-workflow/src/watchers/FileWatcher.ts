import { Watcher, TestConfig } from '../types';
import chokidar from 'chokidar';
import { glob } from 'glob';
import { resolve } from 'path';

export class FileWatcher implements Watcher {
  private config: TestConfig;
  private watcher?: chokidar.FSWatcher;
  private changeCallback?: (files: string[]) => void;
  private debounceTimer?: NodeJS.Timeout;
  private pendingChanges: Set<string> = new Set();

  constructor(config: TestConfig) {
    this.config = config;
  }

  async start(): Promise<void> {
    if (this.watcher) {
      await this.stop();
    }

    // Get all test files to watch
    const testFiles = await this.getTestFiles();
    
    // Get source files that tests might depend on
    const sourceFiles = await this.getSourceFiles();
    
    const filesToWatch = [...testFiles, ...sourceFiles];
    
    if (filesToWatch.length === 0) {
      console.warn('⚠️  No files found to watch');
      return;
    }

    this.watcher = chokidar.watch(filesToWatch, {
      ignored: this.config.watchIgnore || [],
      ignoreInitial: true,
      persistent: true,
      awaitWriteFinish: {
        stabilityThreshold: 100,
        pollInterval: 50
      }
    });

    this.watcher.on('change', (filePath: string) => {
      this.handleFileChange(filePath);
    });

    this.watcher.on('add', (filePath: string) => {
      this.handleFileChange(filePath);
    });

    this.watcher.on('unlink', (filePath: string) => {
      this.handleFileChange(filePath);
    });

    this.watcher.on('error', (error: Error) => {
      console.error('❌ File watcher error:', error);
    });

    this.watcher.on('ready', () => {
      if (!this.config.silent) {
        console.log(`👀 Watching ${filesToWatch.length} files for changes...`);
      }
    });
  }

  async stop(): Promise<void> {
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = undefined;
    }

    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = undefined;
    }

    this.pendingChanges.clear();
  }

  onChange(callback: (files: string[]) => void): void {
    this.changeCallback = callback;
  }

  private handleFileChange(filePath: string): void {
    this.pendingChanges.add(resolve(filePath));

    // Debounce changes to avoid running tests too frequently
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      if (this.changeCallback && this.pendingChanges.size > 0) {
        const changedFiles = Array.from(this.pendingChanges);
        this.pendingChanges.clear();
        this.changeCallback(changedFiles);
      }
    }, 300); // 300ms debounce
  }

  private async getTestFiles(): Promise<string[]> {
    const files: string[] = [];
    
    for (const pattern of this.config.testMatch) {
      try {
        const matches = await glob(pattern, {
          ignore: this.config.testIgnore,
          absolute: true
        });
        files.push(...matches);
      } catch (error) {
        console.warn(`⚠️  Failed to glob pattern ${pattern}:`, error);
      }
    }
    
    return [...new Set(files)]; // Remove duplicates
  }

  private async getSourceFiles(): Promise<string[]> {
    if (!this.config.collectCoverageFrom) {
      return [];
    }

    const files: string[] = [];
    
    for (const pattern of this.config.collectCoverageFrom) {
      try {
        const matches = await glob(pattern, {
          ignore: [
            ...this.config.testIgnore,
            ...this.config.testMatch // Don't watch test files as source files
          ],
          absolute: true
        });
        files.push(...matches);
      } catch (error) {
        console.warn(`⚠️  Failed to glob coverage pattern ${pattern}:`, error);
      }
    }
    
    return [...new Set(files)]; // Remove duplicates
  }
}

export class SmartWatcher extends FileWatcher {
  private dependencyGraph: Map<string, Set<string>> = new Map();

  constructor(config: TestConfig) {
    super(config);
  }

  async start(): Promise<void> {
    await super.start();
    await this.buildDependencyGraph();
  }

  private async buildDependencyGraph(): Promise<void> {
    // This would analyze imports/requires in test files to build a dependency graph
    // For now, we'll use a simple implementation that watches all source files
    const testFiles = await this.getTestFiles();
    const sourceFiles = await this.getSourceFiles();

    // Simple heuristic: associate test files with source files based on naming
    for (const testFile of testFiles) {
      const dependencies = new Set<string>();
      
      // Find corresponding source files
      const testBaseName = testFile
        .replace(/\.(test|spec)\.(js|ts)$/, '')
        .replace(/test[s]?[\/\\]/, '');
      
      for (const sourceFile of sourceFiles) {
        if (sourceFile.includes(testBaseName) || testBaseName.includes(sourceFile.replace(/\.(js|ts)$/, ''))) {
          dependencies.add(sourceFile);
        }
      }
      
      // If no specific dependencies found, depend on all source files
      if (dependencies.size === 0) {
        sourceFiles.forEach(file => dependencies.add(file));
      }
      
      this.dependencyGraph.set(testFile, dependencies);
    }
  }

  protected handleFileChange(filePath: string): void {
    const resolvedPath = resolve(filePath);
    const affectedTests = new Set<string>();

    // Find tests that depend on this file
    for (const [testFile, dependencies] of this.dependencyGraph.entries()) {
      if (dependencies.has(resolvedPath) || testFile === resolvedPath) {
        affectedTests.add(testFile);
      }
    }

    if (affectedTests.size > 0) {
      super.handleFileChange(filePath);
    }
  }

  private async getTestFiles(): Promise<string[]> {
    // Reuse parent method
    return super['getTestFiles']();
  }

  private async getSourceFiles(): Promise<string[]> {
    // Reuse parent method
    return super['getSourceFiles']();
  }
}