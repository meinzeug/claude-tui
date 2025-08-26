import { Worker } from 'worker_threads';
import { resolve } from 'path';
import { TestResult, WorkerMessage } from '../types';
import { EventEmitter } from 'events';

interface WorkerTask {
  id: string;
  testFilePath: string;
  config: any;
  resolve: (result: TestResult) => void;
  reject: (error: Error) => void;
}

export class WorkerPool extends EventEmitter {
  private workers: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private busyWorkers: Set<Worker> = new Set();
  private taskQueue: WorkerTask[] = [];
  private maxWorkers: number;
  private workerPath: string;
  private isShuttingDown: boolean = false;

  constructor(maxWorkers: number = 4) {
    super();
    this.maxWorkers = maxWorkers;
    this.workerPath = resolve(__dirname, './TestWorker.js');
  }

  async initialize(): Promise<void> {
    for (let i = 0; i < this.maxWorkers; i++) {
      await this.createWorker();
    }
  }

  async runTest(testFilePath: string, config: any): Promise<TestResult> {
    return new Promise((resolve, reject) => {
      const task: WorkerTask = {
        id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        testFilePath,
        config,
        resolve,
        reject
      };

      this.taskQueue.push(task);
      this.processQueue();
    });
  }

  private async createWorker(): Promise<Worker> {
    const worker = new Worker(this.workerPath);
    
    worker.on('message', (message: WorkerMessage) => {
      this.handleWorkerMessage(worker, message);
    });

    worker.on('error', (error) => {
      console.error('Worker error:', error);
      this.handleWorkerError(worker, error);
    });

    worker.on('exit', (code) => {
      if (code !== 0 && !this.isShuttingDown) {
        console.error(`Worker exited with code ${code}`);
        this.replaceWorker(worker);
      }
    });

    this.workers.push(worker);
    this.availableWorkers.push(worker);
    
    return worker;
  }

  private handleWorkerMessage(worker: Worker, message: WorkerMessage): void {
    if (message.type === 'result') {
      const result: TestResult = message.payload;
      this.releaseWorker(worker, result);
    } else if (message.type === 'error') {
      const error = new Error(message.payload.message);
      error.stack = message.payload.stack;
      this.releaseWorker(worker, null, error);
    }
  }

  private handleWorkerError(worker: Worker, error: Error): void {
    this.releaseWorker(worker, null, error);
  }

  private releaseWorker(worker: Worker, result?: TestResult | null, error?: Error): void {
    this.busyWorkers.delete(worker);
    this.availableWorkers.push(worker);

    // Find and complete the task
    const currentTask = (worker as any)._currentTask as WorkerTask;
    if (currentTask) {
      if (error) {
        currentTask.reject(error);
      } else if (result) {
        currentTask.resolve(result);
      }
      delete (worker as any)._currentTask;
    }

    // Process next task in queue
    this.processQueue();
  }

  private processQueue(): void {
    if (this.taskQueue.length === 0 || this.availableWorkers.length === 0) {
      return;
    }

    const task = this.taskQueue.shift()!;
    const worker = this.availableWorkers.shift()!;

    this.busyWorkers.add(worker);
    (worker as any)._currentTask = task;

    // Send task to worker
    worker.postMessage({
      type: 'test',
      payload: {
        testFilePath: task.testFilePath,
        config: task.config
      }
    });
  }

  private async replaceWorker(failedWorker: Worker): Promise<void> {
    // Remove failed worker from all arrays
    this.workers = this.workers.filter(w => w !== failedWorker);
    this.availableWorkers = this.availableWorkers.filter(w => w !== failedWorker);
    this.busyWorkers.delete(failedWorker);

    // Handle the task that was running on the failed worker
    const failedTask = (failedWorker as any)._currentTask as WorkerTask;
    if (failedTask) {
      failedTask.reject(new Error('Worker crashed during test execution'));
    }

    // Create a new worker to replace it
    try {
      await this.createWorker();
    } catch (error) {
      console.error('Failed to create replacement worker:', error);
    }
  }

  async shutdown(): Promise<void> {
    this.isShuttingDown = true;

    // Wait for all busy workers to finish
    const busyWorkerPromises = Array.from(this.busyWorkers).map(worker => {
      return new Promise<void>((resolve) => {
        const checkWorker = () => {
          if (!this.busyWorkers.has(worker)) {
            resolve();
          } else {
            setTimeout(checkWorker, 100);
          }
        };
        checkWorker();
      });
    });

    await Promise.all(busyWorkerPromises);

    // Terminate all workers
    const terminationPromises = this.workers.map(worker => {
      return worker.terminate();
    });

    await Promise.all(terminationPromises);

    this.workers = [];
    this.availableWorkers = [];
    this.busyWorkers.clear();
  }

  getStats() {
    return {
      totalWorkers: this.workers.length,
      availableWorkers: this.availableWorkers.length,
      busyWorkers: this.busyWorkers.size,
      queuedTasks: this.taskQueue.length
    };
  }
}