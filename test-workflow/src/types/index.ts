export interface TestConfig {
  testMatch: string[];
  testIgnore: string[];
  setupFiles?: string[];
  setupFilesAfterEnv?: string[];
  collectCoverage?: boolean;
  collectCoverageFrom?: string[];
  coverageDirectory?: string;
  coverageReporters?: string[];
  coverageThreshold?: {
    global?: {
      branches?: number;
      functions?: number;
      lines?: number;
      statements?: number;
    };
  };
  maxWorkers?: number;
  testTimeout?: number;
  verbose?: boolean;
  silent?: boolean;
  bail?: boolean;
  reporters?: ReporterConfig[];
  watchMode?: boolean;
  watchIgnore?: string[];
}

export interface ReporterConfig {
  name: string;
  options?: Record<string, any>;
}

export interface TestResult {
  testFilePath: string;
  success: boolean;
  tests: TestCase[];
  coverage?: CoverageData;
  duration: number;
  error?: string;
}

export interface TestCase {
  name: string;
  success: boolean;
  duration: number;
  error?: string;
  skipped?: boolean;
}

export interface CoverageData {
  lines: {
    total: number;
    covered: number;
    skipped: number;
    pct: number;
  };
  functions: {
    total: number;
    covered: number;
    skipped: number;
    pct: number;
  };
  statements: {
    total: number;
    covered: number;
    skipped: number;
    pct: number;
  };
  branches: {
    total: number;
    covered: number;
    skipped: number;
    pct: number;
  };
}

export interface TestSuite {
  name: string;
  results: TestResult[];
  duration: number;
  passed: number;
  failed: number;
  skipped: number;
  coverage?: CoverageData;
}

export interface Reporter {
  onStart?(config: TestConfig): Promise<void>;
  onTestResult?(result: TestResult): Promise<void>;
  onComplete?(suite: TestSuite): Promise<void>;
}

export interface Watcher {
  start(): Promise<void>;
  stop(): Promise<void>;
  onChange(callback: (files: string[]) => void): void;
}

export interface WorkerMessage {
  type: 'test' | 'result' | 'error';
  payload: any;
}