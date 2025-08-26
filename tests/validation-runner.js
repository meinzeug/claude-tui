#!/usr/bin/env node

/**
 * Simple Validation Runner - No external dependencies
 * Validates MCP server operation and claude-flow integration
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

class ValidationRunner {
  constructor() {
    this.results = {
      mcpTests: [],
      agentTests: [],
      memoryTests: [],
      performanceTests: [],
      startTime: new Date().toISOString()
    };
    this.totalTests = 0;
    this.passedTests = 0;
  }

  async runTest(testName, testFunction) {
    console.log(`🔄 Running: ${testName}`);
    this.totalTests++;
    
    const startTime = Date.now();
    
    try {
      await testFunction();
      const duration = Date.now() - startTime;
      
      console.log(`✅ PASSED: ${testName} (${duration}ms)`);
      this.passedTests++;
      
      return { name: testName, passed: true, duration, error: null };
    } catch (error) {
      const duration = Date.now() - startTime;
      
      console.log(`❌ FAILED: ${testName} (${duration}ms) - ${error.message}`);
      
      return { name: testName, passed: false, duration, error: error.message };
    }
  }

  async validateMCPConnection() {
    return this.runTest('MCP Server Connection', async () => {
      const { stdout } = await execAsync('npx claude-flow@alpha hooks health-check');
      if (!stdout.includes('success') && !stdout.includes('executed')) {
        throw new Error('MCP health check failed');
      }
    });
  }

  async validateSwarmInit() {
    return this.runTest('Swarm Initialization', async () => {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp swarm_init --topology mesh --maxAgents 4');
      const result = JSON.parse(stdout);
      if (!result.success) {
        throw new Error('Swarm initialization failed');
      }
    });
  }

  async validateAgentSpawning() {
    return this.runTest('Agent Spawning', async () => {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp agent_spawn --type tester --name validation-agent');
      const result = JSON.parse(stdout);
      if (!result.success) {
        throw new Error('Agent spawning failed');
      }
    });
  }

  async validateMemoryOperations() {
    return this.runTest('Memory Operations', async () => {
      // Store
      const storeCmd = 'npx claude-flow@alpha mcp memory_usage --action store --key validation-test --value "test-data" --namespace validation';
      const { stdout: storeResult } = await execAsync(storeCmd);
      const store = JSON.parse(storeResult);
      
      if (!store.success) {
        throw new Error('Memory store failed');
      }

      // Retrieve
      const retrieveCmd = 'npx claude-flow@alpha mcp memory_usage --action retrieve --key validation-test --namespace validation';
      const { stdout: retrieveResult } = await execAsync(retrieveCmd);
      const retrieve = JSON.parse(retrieveResult);
      
      if (!retrieve.success) {
        throw new Error('Memory retrieve failed');
      }
    });
  }

  async validateTaskOrchestration() {
    return this.runTest('Task Orchestration', async () => {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp task_orchestrate --task "validation orchestration test" --strategy parallel');
      const result = JSON.parse(stdout);
      if (!result.success) {
        throw new Error('Task orchestration failed');
      }
    });
  }

  async validatePerformanceReporting() {
    return this.runTest('Performance Reporting', async () => {
      const { stdout } = await execAsync('npx claude-flow@alpha mcp performance_report --format summary');
      const result = JSON.parse(stdout);
      if (!result.success) {
        throw new Error('Performance reporting failed');
      }
    });
  }

  async validateConcurrentOperations() {
    return this.runTest('Concurrent Operations', async () => {
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(
          execAsync(`npx claude-flow@alpha mcp health_check --timeout 5000`)
        );
      }
      
      const results = await Promise.allSettled(promises);
      const successCount = results.filter(r => r.status === 'fulfilled').length;
      
      if (successCount < 3) { // At least 60% success rate
        throw new Error(`Concurrent operations failed: only ${successCount}/5 succeeded`);
      }
    });
  }

  async validateErrorRecovery() {
    return this.runTest('Error Recovery', async () => {
      try {
        // Intentionally cause an error
        await execAsync('npx claude-flow@alpha mcp invalid_command');
      } catch (error) {
        // Expected to fail
      }
      
      // Test recovery
      const { stdout } = await execAsync('npx claude-flow@alpha mcp health_check');
      if (!stdout.includes('success') && !stdout.includes('executed')) {
        throw new Error('System did not recover from error');
      }
    });
  }

  async validateAgentCommunication() {
    return this.runTest('Agent Communication', async () => {
      const message = JSON.stringify({ type: 'test_message', data: 'validation' });
      const cmd = `npx claude-flow@alpha mcp daa_communication --from validation-agent --to system --message '${message}'`;
      
      const { stdout } = await execAsync(cmd);
      const result = JSON.parse(stdout);
      
      if (!result.success) {
        throw new Error('Agent communication failed');
      }
    });
  }

  async runAllValidations() {
    console.log('🚀 Starting comprehensive MCP and Claude-Flow validation...\n');
    
    // Core MCP Tests
    this.results.mcpTests.push(await this.validateMCPConnection());
    this.results.mcpTests.push(await this.validateErrorRecovery());
    this.results.mcpTests.push(await this.validateConcurrentOperations());
    
    // Agent and Coordination Tests
    this.results.agentTests.push(await this.validateSwarmInit());
    this.results.agentTests.push(await this.validateAgentSpawning());
    this.results.agentTests.push(await this.validateAgentCommunication());
    this.results.agentTests.push(await this.validateTaskOrchestration());
    
    // Memory and State Tests
    this.results.memoryTests.push(await this.validateMemoryOperations());
    
    // Performance Tests
    this.results.performanceTests.push(await this.validatePerformanceReporting());
    
    this.results.endTime = new Date().toISOString();
    this.results.summary = {
      totalTests: this.totalTests,
      passedTests: this.passedTests,
      failedTests: this.totalTests - this.passedTests,
      successRate: ((this.passedTests / this.totalTests) * 100).toFixed(2)
    };
    
    console.log('\n📊 VALIDATION SUMMARY:');
    console.log(`   Total Tests: ${this.results.summary.totalTests}`);
    console.log(`   Passed: ${this.results.summary.passedTests}`);
    console.log(`   Failed: ${this.results.summary.failedTests}`);
    console.log(`   Success Rate: ${this.results.summary.successRate}%`);
    
    if (this.passedTests === this.totalTests) {
      console.log('\n🎉 ALL VALIDATIONS PASSED! System is 100% operational.');
    } else if (this.passedTests >= this.totalTests * 0.8) {
      console.log('\n✅ SYSTEM MOSTLY OPERATIONAL (≥80% tests passed).');
    } else {
      console.log('\n⚠️  SYSTEM ISSUES DETECTED (<80% tests passed).');
    }
    
    // Save results
    await this.saveResults();
    
    return this.results;
  }

  async saveResults() {
    const tempDir = path.join(__dirname, 'temp');
    await fs.mkdir(tempDir, { recursive: true });
    
    await fs.writeFile(
      path.join(tempDir, 'validation-results.json'),
      JSON.stringify(this.results, null, 2)
    );
    
    // Generate simplified report
    const report = this.generateReport();
    await fs.writeFile(
      path.join(tempDir, 'validation-report.md'),
      report
    );
  }

  generateReport() {
    return `# Hive Mind Validation Report

Generated: ${this.results.endTime}

## Executive Summary
- **Success Rate**: ${this.results.summary.successRate}%
- **Total Tests**: ${this.results.summary.totalTests}
- **Passed**: ${this.results.summary.passedTests}
- **Failed**: ${this.results.summary.failedTests}

## Test Results

### MCP Server Tests
${this.results.mcpTests.map(t => `- ${t.passed ? '✅' : '❌'} ${t.name} (${t.duration}ms)`).join('\n')}

### Agent Coordination Tests  
${this.results.agentTests.map(t => `- ${t.passed ? '✅' : '❌'} ${t.name} (${t.duration}ms)`).join('\n')}

### Memory Management Tests
${this.results.memoryTests.map(t => `- ${t.passed ? '✅' : '❌'} ${t.name} (${t.duration}ms)`).join('\n')}

### Performance Tests
${this.results.performanceTests.map(t => `- ${t.passed ? '✅' : '❌'} ${t.name} (${t.duration}ms)`).join('\n')}

## System Status
${this.results.summary.successRate >= 100 ? '🎉 PERFECT: System is 100% operational' :
  this.results.summary.successRate >= 80 ? '✅ GOOD: System is mostly operational' :
  '⚠️ ISSUES: System needs attention'}

## Recommendations
${this.results.summary.successRate >= 90 ? 
  '- System is operating at optimal levels\n- Continue monitoring performance metrics\n- Regular validation recommended' :
  '- Review failed test details\n- Check system logs for errors\n- Consider system optimization\n- Increase monitoring frequency'}
`;
  }
}

// Run validation if called directly
if (require.main === module) {
  const runner = new ValidationRunner();
  runner.runAllValidations()
    .then(() => process.exit(0))
    .catch(error => {
      console.error('❌ Validation runner failed:', error);
      process.exit(1);
    });
}

module.exports = ValidationRunner;