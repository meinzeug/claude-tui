/**
 * Claude-Flow Integration Test Suite
 * Tests all MCP tools, agent coordination, and memory synchronization
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

describe('Claude-Flow Integration Tests', () => {
  let integrationResults = {
    mcpToolTests: [],
    agentCoordinationTests: [],
    memoryTests: [],
    orchestrationTests: []
  };

  beforeAll(async () => {
    console.log('🚀 Initializing Claude-Flow integration test environment...');
    
    // Ensure temp directory exists
    await fs.mkdir(path.join(__dirname, 'temp'), { recursive: true });
    
    integrationResults.startTime = new Date().toISOString();
    integrationResults.testSuiteVersion = '1.0.0';
  });

  afterAll(async () => {
    console.log('📊 Saving Claude-Flow integration test results...');
    
    integrationResults.endTime = new Date().toISOString();
    await fs.writeFile(
      path.join(__dirname, 'temp', 'integration-test-results.json'),
      JSON.stringify(integrationResults, null, 2)
    );
  });

  describe('MCP Tools Validation', () => {
    const mcpTools = [
      'swarm_init',
      'agent_spawn', 
      'task_orchestrate',
      'swarm_status',
      'agent_list',
      'memory_usage',
      'neural_status',
      'health_check',
      'performance_report',
      'token_usage'
    ];

    test.each(mcpTools)('should validate MCP tool: %s', async (toolName) => {
      const startTime = Date.now();
      let testPassed = false;
      let errorMessage = null;
      let response = null;
      
      try {
        // Test tool execution based on type
        switch(toolName) {
          case 'swarm_init':
            response = await testSwarmInit();
            break;
          case 'agent_spawn':
            response = await testAgentSpawn();
            break;
          case 'task_orchestrate':
            response = await testTaskOrchestrate();
            break;
          case 'swarm_status':
            response = await testSwarmStatus();
            break;
          case 'agent_list':
            response = await testAgentList();
            break;
          case 'memory_usage':
            response = await testMemoryUsage();
            break;
          case 'neural_status':
            response = await testNeuralStatus();
            break;
          case 'health_check':
            response = await testHealthCheck();
            break;
          case 'performance_report':
            response = await testPerformanceReport();
            break;
          case 'token_usage':
            response = await testTokenUsage();
            break;
          default:
            throw new Error(`Unknown tool: ${toolName}`);
        }
        
        testPassed = response && response.success;
        
      } catch (error) {
        errorMessage = error.message;
        console.log(`❌ ${toolName} failed: ${error.message}`);
      }
      
      const executionTime = Date.now() - startTime;
      
      integrationResults.mcpToolTests.push({
        tool: toolName,
        passed: testPassed,
        executionTime,
        response,
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      
      if (testPassed) {
        console.log(`✅ ${toolName} validated successfully (${executionTime}ms)`);
      }
      
      expect(testPassed).toBe(true);
    });

    // Helper functions for tool testing
    async function testSwarmInit() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp swarm_init --topology mesh --maxAgents 4');
      return JSON.parse(stdout);
    }

    async function testAgentSpawn() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp agent_spawn --type tester --name integration-tester');
      return JSON.parse(stdout);
    }

    async function testTaskOrchestrate() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp task_orchestrate --task "integration test validation" --strategy parallel');
      return JSON.parse(stdout);
    }

    async function testSwarmStatus() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp swarm_status');
      return JSON.parse(stdout);
    }

    async function testAgentList() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp agent_list');
      return JSON.parse(stdout);
    }

    async function testMemoryUsage() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp memory_usage --action store --key test-key --value "integration test data"');
      return JSON.parse(stdout);
    }

    async function testNeuralStatus() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp neural_status');
      return JSON.parse(stdout);
    }

    async function testHealthCheck() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp health_check');
      return JSON.parse(stdout);
    }

    async function testPerformanceReport() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp performance_report --format summary');
      return JSON.parse(stdout);
    }

    async function testTokenUsage() {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp token_usage --operation integration-test');
      return JSON.parse(stdout);
    }
  });

  describe('Agent Coordination Tests', () => {
    test('should coordinate multiple agents successfully', async () => {
      const testStartTime = Date.now();
      
      try {
        // Initialize swarm for coordination test
        await execAsync('npx claude-flow@alpha mcp swarm_init --topology hierarchical --maxAgents 6');
        
        // Spawn multiple agents
        const agents = [
          { type: 'coordinator', name: 'test-coordinator' },
          { type: 'analyst', name: 'test-analyst' },
          { type: 'tester', name: 'test-tester' },
          { type: 'monitor', name: 'test-monitor' }
        ];
        
        const spawnedAgents = [];
        for (const agent of agents) {
          const { stdout } = await execAsync(`npx claude-flow@alpha mcp agent_spawn --type ${agent.type} --name ${agent.name}`);
          const spawnResult = JSON.parse(stdout);
          spawnedAgents.push(spawnResult);
        }
        
        // Test coordination sync
        const { stdout: syncResult } = await execAsync('npx claude-flow@alpha mcp coordination_sync');
        const coordinationResult = JSON.parse(syncResult);
        
        // Verify all agents are coordinated
        const { stdout: statusResult } = await execAsync('npx claude-flow@alpha mcp swarm_status');
        const swarmStatus = JSON.parse(statusResult);
        
        const coordinationTest = {
          test: 'multi_agent_coordination',
          passed: coordinationResult.success && swarmStatus.agentCount >= 4,
          agentsSpawned: spawnedAgents.length,
          coordinationSuccessful: coordinationResult.success,
          swarmStatus,
          executionTime: Date.now() - testStartTime,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.agentCoordinationTests.push(coordinationTest);
        
        expect(coordinationTest.passed).toBe(true);
        console.log(`✅ Agent coordination test passed: ${spawnedAgents.length} agents coordinated`);
        
      } catch (error) {
        const failedTest = {
          test: 'multi_agent_coordination',
          passed: false,
          error: error.message,
          executionTime: Date.now() - testStartTime,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.agentCoordinationTests.push(failedTest);
        throw error;
      }
    });

    test('should handle agent communication', async () => {
      try {
        // Test inter-agent communication
        const { stdout } = await execAsync('npx claude-flow@alpha mcp daa_communication --from test-coordinator --to test-analyst --message "{\\"type\\": \\"coordination_request\\", \\"task\\": \\"analysis\\"}"');
        const commResult = JSON.parse(stdout);
        
        const communicationTest = {
          test: 'agent_communication',
          passed: commResult.success,
          communicationResult: commResult,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.agentCoordinationTests.push(communicationTest);
        
        expect(commResult.success).toBe(true);
        console.log('✅ Agent communication test passed');
        
      } catch (error) {
        console.log(`❌ Agent communication failed: ${error.message}`);
        throw error;
      }
    });
  });

  describe('Memory Synchronization Tests', () => {
    test('should synchronize memory across agents', async () => {
      const testData = {
        testKey1: 'coordination-data-1',
        testKey2: 'coordination-data-2',
        testKey3: 'shared-context-data'
      };
      
      try {
        // Store data in different namespaces
        for (const [key, value] of Object.entries(testData)) {
          await execAsync(`npx claude-flow@alpha mcp memory_usage --action store --key ${key} --value "${value}" --namespace coordination-test`);
        }
        
        // Test memory search functionality
        const { stdout: searchResult } = await execAsync('npx claude-flow@alpha mcp memory_search --pattern "coordination-data" --namespace coordination-test --limit 10');
        const searchData = JSON.parse(searchResult);
        
        // Test memory retrieval
        const { stdout: retrieveResult } = await execAsync('npx claude-flow@alpha mcp memory_usage --action retrieve --key testKey1 --namespace coordination-test');
        const retrieveData = JSON.parse(retrieveResult);
        
        // Test memory sync across instances
        const { stdout: syncResult } = await execAsync('npx claude-flow@alpha mcp memory_sync --target coordination-test');
        const syncData = JSON.parse(syncResult);
        
        const memoryTest = {
          test: 'memory_synchronization',
          passed: searchData.success && retrieveData.success && syncData.success,
          itemsStored: Object.keys(testData).length,
          searchResults: searchData,
          retrieveResults: retrieveData,
          syncResults: syncData,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.memoryTests.push(memoryTest);
        
        expect(memoryTest.passed).toBe(true);
        console.log(`✅ Memory synchronization test passed: ${memoryTest.itemsStored} items managed`);
        
      } catch (error) {
        const failedTest = {
          test: 'memory_synchronization',
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.memoryTests.push(failedTest);
        throw error;
      }
    });

    test('should handle memory persistence across sessions', async () => {
      try {
        // Test cross-session persistence
        const { stdout: persistResult } = await execAsync('npx claude-flow@alpha mcp memory_persist --sessionId integration-test-session');
        const persistData = JSON.parse(persistResult);
        
        // Test state snapshots
        const { stdout: snapshotResult } = await execAsync('npx claude-flow@alpha mcp state_snapshot --name integration-test-snapshot');
        const snapshotData = JSON.parse(snapshotResult);
        
        const persistenceTest = {
          test: 'memory_persistence',
          passed: persistData.success && snapshotData.success,
          sessionPersistence: persistData,
          snapshotCreation: snapshotData,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.memoryTests.push(persistenceTest);
        
        expect(persistenceTest.passed).toBe(true);
        console.log('✅ Memory persistence test passed');
        
      } catch (error) {
        console.log(`❌ Memory persistence failed: ${error.message}`);
        throw error;
      }
    });
  });

  describe('Orchestration Tests', () => {
    test('should orchestrate parallel task execution', async () => {
      try {
        // Test parallel task execution
        const tasks = [
          'analysis-task-1',
          'validation-task-2', 
          'optimization-task-3',
          'monitoring-task-4'
        ];
        
        const { stdout: parallelResult } = await execAsync(`npx claude-flow@alpha mcp parallel_execute --tasks '${JSON.stringify(tasks)}'`);
        const parallelData = JSON.parse(parallelResult);
        
        // Test load balancing
        const { stdout: balanceResult } = await execAsync(`npx claude-flow@alpha mcp load_balance --tasks '${JSON.stringify(tasks)}'`);
        const balanceData = JSON.parse(balanceResult);
        
        const orchestrationTest = {
          test: 'parallel_orchestration',
          passed: parallelData.success && balanceData.success,
          tasksExecuted: tasks.length,
          parallelExecution: parallelData,
          loadBalancing: balanceData,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.orchestrationTests.push(orchestrationTest);
        
        expect(orchestrationTest.passed).toBe(true);
        console.log(`✅ Parallel orchestration test passed: ${tasks.length} tasks executed`);
        
      } catch (error) {
        const failedTest = {
          test: 'parallel_orchestration',
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.orchestrationTests.push(failedTest);
        throw error;
      }
    });

    test('should handle consensus mechanisms', async () => {
      try {
        // Test consensus building
        const proposal = {
          type: 'coordination_strategy',
          strategy: 'hierarchical',
          parameters: { maxDepth: 3, branching: 4 }
        };
        
        const { stdout: consensusResult } = await execAsync(`npx claude-flow@alpha mcp daa_consensus --agents '["test-coordinator", "test-analyst", "test-monitor"]' --proposal '${JSON.stringify(proposal)}'`);
        const consensusData = JSON.parse(consensusResult);
        
        const consensusTest = {
          test: 'consensus_mechanism',
          passed: consensusData.success,
          proposal,
          consensusResult: consensusData,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.orchestrationTests.push(consensusTest);
        
        expect(consensusTest.passed).toBe(true);
        console.log('✅ Consensus mechanism test passed');
        
      } catch (error) {
        console.log(`❌ Consensus mechanism failed: ${error.message}`);
        throw error;
      }
    });
  });

  describe('System Integration Tests', () => {
    test('should validate complete system integration', async () => {
      const integrationStartTime = Date.now();
      
      try {
        // Full system integration test
        const systemTests = [
          async () => {
            const { stdout } = await execAsync('npx claude-flow@alpha mcp health_check');
            return JSON.parse(stdout);
          },
          async () => {
            const { stdout } = await execAsync('npx claude-flow@alpha mcp swarm_status');
            return JSON.parse(stdout);
          },
          async () => {
            const { stdout } = await execAsync('npx claude-flow@alpha mcp agent_list');
            return JSON.parse(stdout);
          },
          async () => {
            const { stdout } = await execAsync('npx claude-flow@alpha mcp performance_report --format summary');
            return JSON.parse(stdout);
          }
        ];
        
        const systemResults = await Promise.all(systemTests.map(test => test()));
        const allSuccessful = systemResults.every(result => result.success);
        
        const systemIntegrationTest = {
          test: 'complete_system_integration',
          passed: allSuccessful,
          testsRun: systemTests.length,
          systemResults,
          executionTime: Date.now() - integrationStartTime,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.orchestrationTests.push(systemIntegrationTest);
        
        expect(allSuccessful).toBe(true);
        console.log(`✅ Complete system integration test passed: ${systemTests.length} components validated`);
        
      } catch (error) {
        const failedTest = {
          test: 'complete_system_integration',
          passed: false,
          error: error.message,
          executionTime: Date.now() - integrationStartTime,
          timestamp: new Date().toISOString()
        };
        
        integrationResults.orchestrationTests.push(failedTest);
        throw error;
      }
    });
  });
});