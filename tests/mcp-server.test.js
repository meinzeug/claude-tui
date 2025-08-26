/**
 * MCP Server Validation Test Suite
 * Tests connection stability, error recovery, and background execution
 */

const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

describe('MCP Server Validation', () => {
  let serverProcess;
  let testResults = {
    connectionTests: [],
    stabilityTests: [],
    errorRecoveryTests: [],
    backgroundExecutionTests: []
  };

  beforeAll(async () => {
    console.log('🔧 Setting up MCP server validation environment...');
    
    // Ensure test directory exists
    await fs.mkdir(path.join(__dirname, 'temp'), { recursive: true });
    
    // Initialize test metrics
    testResults.startTime = new Date().toISOString();
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up MCP server test environment...');
    
    if (serverProcess) {
      serverProcess.kill();
    }
    
    // Save test results
    testResults.endTime = new Date().toISOString();
    await fs.writeFile(
      path.join(__dirname, 'temp', 'mcp-server-test-results.json'),
      JSON.stringify(testResults, null, 2)
    );
  });

  describe('Connection Stability Tests', () => {
    test('should establish MCP connection successfully', async () => {
      const startTime = Date.now();
      
      try {
        const { stdout, stderr } = await execAsync('npx claude-flow@alpha hooks health-check');
        
        const connectionTime = Date.now() - startTime;
        
        expect(stdout).toContain('success');
        expect(connectionTime).toBeLessThan(5000); // Should connect within 5 seconds
        
        testResults.connectionTests.push({
          test: 'basic_connection',
          passed: true,
          connectionTime,
          timestamp: new Date().toISOString()
        });
        
        console.log(`✅ MCP connection established in ${connectionTime}ms`);
      } catch (error) {
        testResults.connectionTests.push({
          test: 'basic_connection',
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
        throw error;
      }
    }, 10000);

    test('should handle concurrent connections', async () => {
      const concurrentRequests = 5;
      const promises = [];
      
      for (let i = 0; i < concurrentRequests; i++) {
        promises.push(execAsync(`npx claude-flow@alpha hooks ping --id connection-${i}`));
      }
      
      const startTime = Date.now();
      
      try {
        const results = await Promise.allSettled(promises);
        const totalTime = Date.now() - startTime;
        
        const successCount = results.filter(r => r.status === 'fulfilled').length;
        const successRate = (successCount / concurrentRequests) * 100;
        
        expect(successRate).toBeGreaterThan(80); // At least 80% success rate
        
        testResults.connectionTests.push({
          test: 'concurrent_connections',
          passed: true,
          concurrentRequests,
          successCount,
          successRate,
          totalTime,
          timestamp: new Date().toISOString()
        });
        
        console.log(`✅ Concurrent connections: ${successCount}/${concurrentRequests} (${successRate}%)`);
      } catch (error) {
        testResults.connectionTests.push({
          test: 'concurrent_connections',
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
        throw error;
      }
    }, 15000);

    test('should maintain connection over time', async () => {
      const testDuration = 30000; // 30 seconds
      const pingInterval = 2000; // 2 seconds
      const startTime = Date.now();
      let pingCount = 0;
      let successCount = 0;
      
      const pingTest = setInterval(async () => {
        pingCount++;
        
        try {
          await execAsync('npx claude-flow@alpha hooks ping --timeout 1000');
          successCount++;
        } catch (error) {
          console.log(`⚠️ Ping ${pingCount} failed: ${error.message}`);
        }
        
        if (Date.now() - startTime >= testDuration) {
          clearInterval(pingTest);
          
          const uptime = (successCount / pingCount) * 100;
          
          testResults.stabilityTests.push({
            test: 'connection_stability',
            passed: uptime > 90,
            pingCount,
            successCount,
            uptimePercentage: uptime,
            duration: testDuration,
            timestamp: new Date().toISOString()
          });
          
          expect(uptime).toBeGreaterThan(90); // 90% uptime required
          console.log(`✅ Connection stability: ${uptime.toFixed(2)}% uptime`);
        }
      }, pingInterval);
      
      // Wait for test to complete
      await new Promise(resolve => setTimeout(resolve, testDuration + 1000));
    }, 35000);
  });

  describe('Error Recovery Tests', () => {
    test('should recover from timeout errors', async () => {
      try {
        // Simulate timeout by using very short timeout
        await execAsync('npx claude-flow@alpha hooks ping --timeout 1');
      } catch (initialError) {
        // Now test recovery
        const { stdout } = await execAsync('npx claude-flow@alpha hooks ping --timeout 5000');
        
        expect(stdout).toContain('success');
        
        testResults.errorRecoveryTests.push({
          test: 'timeout_recovery',
          passed: true,
          initialError: initialError.message,
          recoverySuccessful: true,
          timestamp: new Date().toISOString()
        });
        
        console.log('✅ Successfully recovered from timeout error');
      }
    });

    test('should handle malformed requests gracefully', async () => {
      const malformedRequests = [
        'npx claude-flow@alpha hooks invalid-command',
        'npx claude-flow@alpha hooks ping --invalid-flag',
        'npx claude-flow@alpha hooks "" --empty-command'
      ];
      
      let gracefulFailures = 0;
      
      for (const request of malformedRequests) {
        try {
          await execAsync(request);
        } catch (error) {
          // Should fail gracefully without crashing server
          if (error.code !== 0 && !error.message.includes('ECONNREFUSED')) {
            gracefulFailures++;
          }
        }
      }
      
      // Test that server is still responsive after malformed requests
      const { stdout } = await execAsync('npx claude-flow@alpha hooks ping');
      expect(stdout).toContain('success');
      
      testResults.errorRecoveryTests.push({
        test: 'malformed_request_handling',
        passed: true,
        malformedRequestsCount: malformedRequests.length,
        gracefulFailures,
        serverStillResponsive: true,
        timestamp: new Date().toISOString()
      });
      
      console.log(`✅ Handled ${gracefulFailures}/${malformedRequests.length} malformed requests gracefully`);
    });

    test('should handle resource exhaustion', async () => {
      const resourceTestPromises = [];
      
      // Create many simultaneous requests to test resource limits
      for (let i = 0; i < 20; i++) {
        resourceTestPromises.push(
          execAsync(`npx claude-flow@alpha hooks ping --id resource-test-${i}`)
            .catch(error => ({ error: error.message, id: i }))
        );
      }
      
      const results = await Promise.allSettled(resourceTestPromises);
      const successful = results.filter(r => r.status === 'fulfilled' && !r.value.error).length;
      
      // After resource test, ensure server is still responsive
      const { stdout } = await execAsync('npx claude-flow@alpha hooks health-check');
      expect(stdout).toContain('success');
      
      testResults.errorRecoveryTests.push({
        test: 'resource_exhaustion_recovery',
        passed: true,
        totalRequests: 20,
        successfulRequests: successful,
        serverRecovered: true,
        timestamp: new Date().toISOString()
      });
      
      console.log(`✅ Resource exhaustion test: ${successful}/20 requests successful, server recovered`);
    });
  });

  describe('Background Execution Tests', () => {
    test('should support background task execution', async () => {
      // Start a long-running background task
      const backgroundTask = spawn('npx', ['claude-flow@alpha', 'hooks', 'background-ping', '--duration', '10000']);
      
      let backgroundOutput = '';
      let errorOutput = '';
      
      backgroundTask.stdout.on('data', (data) => {
        backgroundOutput += data.toString();
      });
      
      backgroundTask.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      // Allow background task to start
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Test that we can still make other requests while background task runs
      const { stdout } = await execAsync('npx claude-flow@alpha hooks ping --id foreground-during-background');
      expect(stdout).toContain('success');
      
      // Stop background task
      backgroundTask.kill();
      
      await new Promise(resolve => {
        backgroundTask.on('close', (code) => {
          testResults.backgroundExecutionTests.push({
            test: 'background_task_support',
            passed: true,
            backgroundTaskOutput: backgroundOutput.substring(0, 500), // First 500 chars
            foregroundTaskSuccessful: stdout.includes('success'),
            backgroundTaskTerminated: true,
            timestamp: new Date().toISOString()
          });
          
          console.log('✅ Background execution supported, foreground operations unblocked');
          resolve();
        });
      });
    }, 15000);

    test('should handle multiple background tasks', async () => {
      const backgroundTasks = [];
      const taskCount = 3;
      
      // Start multiple background tasks
      for (let i = 0; i < taskCount; i++) {
        const task = spawn('npx', ['claude-flow@alpha', 'hooks', 'background-ping', '--id', `bg-${i}`, '--duration', '5000']);
        backgroundTasks.push(task);
      }
      
      // Allow tasks to start
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Test foreground operation while multiple background tasks run
      const { stdout } = await execAsync('npx claude-flow@alpha hooks ping --id foreground-multi-bg');
      expect(stdout).toContain('success');
      
      // Clean up background tasks
      backgroundTasks.forEach(task => task.kill());
      
      testResults.backgroundExecutionTests.push({
        test: 'multiple_background_tasks',
        passed: true,
        backgroundTaskCount: taskCount,
        foregroundOperationSuccessful: true,
        timestamp: new Date().toISOString()
      });
      
      console.log(`✅ Multiple background tasks (${taskCount}) handled successfully`);
    });
  });

  describe('Performance Benchmarks', () => {
    test('should meet response time requirements', async () => {
      const iterations = 10;
      const responseTimes = [];
      
      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        await execAsync('npx claude-flow@alpha hooks ping');
        const responseTime = Date.now() - startTime;
        responseTimes.push(responseTime);
      }
      
      const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / iterations;
      const maxResponseTime = Math.max(...responseTimes);
      const minResponseTime = Math.min(...responseTimes);
      
      expect(avgResponseTime).toBeLessThan(1000); // Average < 1 second
      expect(maxResponseTime).toBeLessThan(2000); // Max < 2 seconds
      
      testResults.stabilityTests.push({
        test: 'response_time_benchmark',
        passed: avgResponseTime < 1000 && maxResponseTime < 2000,
        iterations,
        avgResponseTime,
        maxResponseTime,
        minResponseTime,
        responseTimes,
        timestamp: new Date().toISOString()
      });
      
      console.log(`✅ Performance: avg ${avgResponseTime.toFixed(2)}ms, max ${maxResponseTime}ms, min ${minResponseTime}ms`);
    });
  });
});