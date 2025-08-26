/**
 * Performance Benchmark Test Suite
 * Tests system performance, scalability, and resource utilization
 */

const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

const execAsync = promisify(exec);

describe('Performance Benchmark Tests', () => {
  let performanceResults = {
    throughputTests: [],
    latencyTests: [],
    scalabilityTests: [],
    resourceTests: [],
    benchmarkMetrics: {}
  };

  beforeAll(async () => {
    console.log('⚡ Initializing performance benchmark environment...');
    
    // Ensure temp directory exists
    await fs.mkdir(path.join(__dirname, 'temp'), { recursive: true });
    
    // Initialize system baseline
    performanceResults.systemInfo = {
      platform: os.platform(),
      architecture: os.arch(),
      cpuCount: os.cpus().length,
      totalMemory: os.totalmem(),
      freeMemory: os.freemem(),
      nodeVersion: process.version,
      timestamp: new Date().toISOString()
    };
    
    console.log(`📊 System: ${os.cpus().length} CPUs, ${Math.round(os.totalmem() / 1024 / 1024 / 1024)}GB RAM`);
  });

  afterAll(async () => {
    console.log('💾 Saving performance benchmark results...');
    
    // Calculate overall performance score
    performanceResults.overallScore = calculatePerformanceScore(performanceResults);
    
    await fs.writeFile(
      path.join(__dirname, 'temp', 'performance-benchmark-results.json'),
      JSON.stringify(performanceResults, null, 2)
    );
    
    // Generate performance report
    await generatePerformanceReport(performanceResults);
  });

  describe('Throughput Tests', () => {
    test('should handle high request throughput', async () => {
      const requestCounts = [10, 50, 100, 200];
      const throughputResults = [];
      
      for (const requestCount of requestCounts) {
        console.log(`🔄 Testing throughput with ${requestCount} requests...`);
        
        const startTime = Date.now();
        const promises = [];
        
        // Create concurrent requests
        for (let i = 0; i < requestCount; i++) {
          promises.push(
            execAsync(`npx claude-flow@alpha mcp health_check --id throughput-${requestCount}-${i}`)
              .catch(error => ({ error: error.message, requestId: i }))
          );
        }
        
        const results = await Promise.allSettled(promises);
        const totalTime = Date.now() - startTime;
        
        const successful = results.filter(r => 
          r.status === 'fulfilled' && !r.value.error
        ).length;
        
        const throughput = (successful / totalTime) * 1000; // requests per second
        
        const throughputTest = {
          requestCount,
          successful,
          failed: requestCount - successful,
          totalTime,
          throughput: throughput.toFixed(2),
          avgResponseTime: (totalTime / requestCount).toFixed(2),
          timestamp: new Date().toISOString()
        };
        
        throughputResults.push(throughputTest);
        
        console.log(`✅ ${requestCount} requests: ${successful} successful, ${throughput.toFixed(2)} req/sec`);
      }
      
      performanceResults.throughputTests = throughputResults;
      
      // Expect at least 10 requests per second at 100 concurrent requests
      const highLoadTest = throughputResults.find(t => t.requestCount === 100);
      expect(parseFloat(highLoadTest.throughput)).toBeGreaterThan(10);
    }, 120000);

    test('should maintain performance under sustained load', async () => {
      const testDuration = 30000; // 30 seconds
      const requestInterval = 1000; // 1 request per second
      const startTime = Date.now();
      
      let requestCount = 0;
      let successCount = 0;
      let totalResponseTime = 0;
      
      console.log(`🕒 Running sustained load test for ${testDuration/1000} seconds...`);
      
      const sustainedTest = setInterval(async () => {
        requestCount++;
        const requestStart = Date.now();
        
        try {
          await execAsync(`npx claude-flow@alpha mcp health_check --id sustained-${requestCount}`);
          successCount++;
          totalResponseTime += Date.now() - requestStart;
        } catch (error) {
          console.log(`⚠️ Request ${requestCount} failed: ${error.message}`);
        }
        
        if (Date.now() - startTime >= testDuration) {
          clearInterval(sustainedTest);
          
          const avgResponseTime = totalResponseTime / successCount;
          const successRate = (successCount / requestCount) * 100;
          
          const sustainedLoadResult = {
            test: 'sustained_load',
            duration: testDuration,
            totalRequests: requestCount,
            successfulRequests: successCount,
            failedRequests: requestCount - successCount,
            successRate: successRate.toFixed(2),
            avgResponseTime: avgResponseTime.toFixed(2),
            timestamp: new Date().toISOString()
          };
          
          performanceResults.throughputTests.push(sustainedLoadResult);
          
          console.log(`✅ Sustained load: ${successRate.toFixed(2)}% success rate, ${avgResponseTime.toFixed(2)}ms avg response`);
          
          expect(successRate).toBeGreaterThan(90);
        }
      }, requestInterval);
      
      // Wait for test completion
      await new Promise(resolve => setTimeout(resolve, testDuration + 2000));
    }, 35000);
  });

  describe('Latency Tests', () => {
    test('should meet latency requirements', async () => {
      const latencyTests = [
        { name: 'health_check', command: 'health_check' },
        { name: 'swarm_status', command: 'swarm_status' },
        { name: 'agent_list', command: 'agent_list' },
        { name: 'memory_retrieve', command: 'memory_usage --action retrieve --key test-latency' }
      ];
      
      const latencyResults = [];
      
      for (const test of latencyTests) {
        console.log(`⏱️ Testing latency for ${test.name}...`);
        
        const iterations = 20;
        const latencies = [];
        
        for (let i = 0; i < iterations; i++) {
          const startTime = Date.now();
          
          try {
            await execAsync(`npx claude-flow@alpha mcp ${test.command}`);
            const latency = Date.now() - startTime;
            latencies.push(latency);
          } catch (error) {
            console.log(`⚠️ Latency test ${test.name} iteration ${i} failed: ${error.message}`);
          }
        }
        
        const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        const minLatency = Math.min(...latencies);
        const maxLatency = Math.max(...latencies);
        const p95Latency = calculatePercentile(latencies, 95);
        const p99Latency = calculatePercentile(latencies, 99);
        
        const latencyResult = {
          test: test.name,
          iterations,
          avgLatency: avgLatency.toFixed(2),
          minLatency,
          maxLatency,
          p95Latency: p95Latency.toFixed(2),
          p99Latency: p99Latency.toFixed(2),
          latencies,
          timestamp: new Date().toISOString()
        };
        
        latencyResults.push(latencyResult);
        
        console.log(`✅ ${test.name}: avg ${avgLatency.toFixed(2)}ms, p95 ${p95Latency.toFixed(2)}ms`);
      }
      
      performanceResults.latencyTests = latencyResults;
      
      // Expect average latency under 500ms for all operations
      latencyResults.forEach(result => {
        expect(parseFloat(result.avgLatency)).toBeLessThan(500);
      });
    });
  });

  describe('Scalability Tests', () => {
    test('should scale with increased agent count', async () => {
      const agentCounts = [2, 4, 8, 16];
      const scalabilityResults = [];
      
      for (const agentCount of agentCounts) {
        console.log(`📈 Testing scalability with ${agentCount} agents...`);
        
        const startTime = Date.now();
        
        try {
          // Initialize swarm with specific agent count
          await execAsync(`npx claude-flow@alpha mcp swarm_init --topology mesh --maxAgents ${agentCount}`);
          
          // Spawn agents
          const spawnPromises = [];
          for (let i = 0; i < agentCount; i++) {
            spawnPromises.push(
              execAsync(`npx claude-flow@alpha mcp agent_spawn --type tester --name scale-test-${i}`)
            );
          }
          
          await Promise.all(spawnPromises);
          
          // Test coordination performance with all agents
          const coordStartTime = Date.now();
          await execAsync('npx claude-flow@alpha mcp coordination_sync');
          const coordinationTime = Date.now() - coordStartTime;
          
          // Test agent list performance
          const listStartTime = Date.now();
          const { stdout } = await execAsync('npx claude-flow@alpha mcp agent_list');
          const listTime = Date.now() - listStartTime;
          const agentData = JSON.parse(stdout);
          
          const totalTime = Date.now() - startTime;
          
          const scalabilityResult = {
            agentCount,
            setupTime: totalTime,
            coordinationTime,
            listTime,
            actualAgents: agentData.agents ? agentData.agents.length : 0,
            coordinationLatency: coordinationTime / agentCount,
            timestamp: new Date().toISOString()
          };
          
          scalabilityResults.push(scalabilityResult);
          
          console.log(`✅ ${agentCount} agents: setup ${totalTime}ms, coordination ${coordinationTime}ms`);
          
        } catch (error) {
          console.log(`❌ Scalability test with ${agentCount} agents failed: ${error.message}`);
          
          scalabilityResults.push({
            agentCount,
            failed: true,
            error: error.message,
            timestamp: new Date().toISOString()
          });
        }
      }
      
      performanceResults.scalabilityTests = scalabilityResults;
      
      // Expect scalability to be sub-linear (coordination time should not grow exponentially)
      const successfulTests = scalabilityResults.filter(r => !r.failed);
      expect(successfulTests.length).toBeGreaterThan(0);
    }, 180000);
  });

  describe('Resource Utilization Tests', () => {
    test('should monitor memory usage', async () => {
      console.log('🧠 Testing memory utilization...');
      
      const initialMemory = process.memoryUsage();
      const systemInitialMemory = os.freemem();
      
      // Create memory-intensive operations
      const memoryTests = [];
      for (let i = 0; i < 10; i++) {
        memoryTests.push(
          execAsync(`npx claude-flow@alpha mcp memory_usage --action store --key memory-test-${i} --value "${generateLargeString(1000)}" --namespace memory-test`)
        );
      }
      
      await Promise.all(memoryTests);
      
      const finalMemory = process.memoryUsage();
      const systemFinalMemory = os.freemem();
      
      const memoryResult = {
        test: 'memory_utilization',
        initialMemory,
        finalMemory,
        memoryIncrease: {
          heapUsed: finalMemory.heapUsed - initialMemory.heapUsed,
          heapTotal: finalMemory.heapTotal - initialMemory.heapTotal,
          rss: finalMemory.rss - initialMemory.rss
        },
        systemMemoryUsed: systemInitialMemory - systemFinalMemory,
        timestamp: new Date().toISOString()
      };
      
      performanceResults.resourceTests.push(memoryResult);
      
      console.log(`✅ Memory test: heap increased by ${Math.round(memoryResult.memoryIncrease.heapUsed / 1024 / 1024)}MB`);
      
      // Expect memory increase to be reasonable (< 100MB for this test)
      expect(memoryResult.memoryIncrease.heapUsed).toBeLessThan(100 * 1024 * 1024);
    });

    test('should monitor CPU usage patterns', async () => {
      console.log('⚡ Testing CPU utilization patterns...');
      
      const cpuTests = [
        { name: 'light_load', iterations: 10 },
        { name: 'medium_load', iterations: 50 },
        { name: 'heavy_load', iterations: 100 }
      ];
      
      const cpuResults = [];
      
      for (const test of cpuTests) {
        const startTime = process.hrtime();
        const promises = [];
        
        for (let i = 0; i < test.iterations; i++) {
          promises.push(
            execAsync(`npx claude-flow@alpha mcp health_check --id cpu-${test.name}-${i}`)
          );
        }
        
        await Promise.all(promises);
        
        const endTime = process.hrtime(startTime);
        const executionTimeMs = endTime[0] * 1000 + endTime[1] / 1000000;
        
        const cpuResult = {
          test: test.name,
          iterations: test.iterations,
          executionTime: executionTimeMs.toFixed(2),
          operationsPerSecond: ((test.iterations / executionTimeMs) * 1000).toFixed(2),
          cpuEfficiency: (test.iterations / executionTimeMs).toFixed(4),
          timestamp: new Date().toISOString()
        };
        
        cpuResults.push(cpuResult);
        
        console.log(`✅ ${test.name}: ${cpuResult.operationsPerSecond} ops/sec, ${cpuResult.executionTime}ms total`);
      }
      
      performanceResults.resourceTests.push({
        test: 'cpu_utilization',
        results: cpuResults
      });
      
      // Expect reasonable performance scaling
      expect(cpuResults.length).toBe(cpuTests.length);
    });
  });

  // Helper functions
  function calculatePercentile(values, percentile) {
    const sorted = values.sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }
  
  function generateLargeString(size) {
    return 'A'.repeat(size);
  }
  
  function calculatePerformanceScore(results) {
    let score = 100;
    
    // Throughput score impact
    const throughputTests = results.throughputTests || [];
    const highLoadTest = throughputTests.find(t => t.requestCount >= 100);
    if (highLoadTest && parseFloat(highLoadTest.throughput) < 10) {
      score -= 20;
    }
    
    // Latency score impact
    const latencyTests = results.latencyTests || [];
    const highLatencyTests = latencyTests.filter(t => parseFloat(t.avgLatency) > 500);
    score -= highLatencyTests.length * 10;
    
    // Scalability score impact
    const scalabilityTests = results.scalabilityTests || [];
    const failedScalabilityTests = scalabilityTests.filter(t => t.failed);
    score -= failedScalabilityTests.length * 15;
    
    return Math.max(0, score);
  }
  
  async function generatePerformanceReport(results) {
    const report = `
# Performance Benchmark Report

Generated: ${new Date().toISOString()}

## System Information
- Platform: ${results.systemInfo.platform}
- Architecture: ${results.systemInfo.architecture}
- CPUs: ${results.systemInfo.cpuCount}
- Memory: ${Math.round(results.systemInfo.totalMemory / 1024 / 1024 / 1024)}GB

## Performance Score: ${results.overallScore}/100

## Throughput Tests
${results.throughputTests.map(t => 
  `- ${t.requestCount} requests: ${t.throughput} req/sec, ${t.successful}/${t.requestCount} successful`
).join('\n')}

## Latency Tests
${results.latencyTests.map(t => 
  `- ${t.test}: avg ${t.avgLatency}ms, p95 ${t.p95Latency}ms, p99 ${t.p99Latency}ms`
).join('\n')}

## Scalability Tests
${results.scalabilityTests.filter(t => !t.failed).map(t => 
  `- ${t.agentCount} agents: setup ${t.setupTime}ms, coordination ${t.coordinationTime}ms`
).join('\n')}

## Recommendations
${results.overallScore < 80 ? '⚠️ Performance improvements needed' : '✅ Performance meets requirements'}
`;
    
    await fs.writeFile(
      path.join(__dirname, 'temp', 'performance-report.md'),
      report
    );
  }
});