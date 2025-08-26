#!/usr/bin/env node

/**
 * MCP Server Health Check - Production Monitoring
 * Comprehensive health monitoring for claude-flow MCP server
 */

const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');
const { performance } = require('perf_hooks');

class MCPHealthMonitor {
    constructor() {
        this.config = {
            pidFile: '/home/tekkadmin/claude-tui/scripts/mcp/mcp-server.pid',
            logDir: '/home/tekkadmin/claude-tui/logs/mcp',
            healthLogFile: '/home/tekkadmin/claude-tui/logs/mcp/health-check.log',
            maxMemoryMB: 512,
            maxCpuPercent: 80,
            responseTimeoutMs: 5000,
            healthInterval: 30000,
            alertThresholds: {
                memory: { warning: 256, critical: 450 },
                cpu: { warning: 60, critical: 80 },
                responseTime: { warning: 2000, critical: 4000 }
            }
        };
        
        this.metrics = {
            startTime: Date.now(),
            checks: 0,
            failures: 0,
            lastCheck: null,
            averageResponseTime: 0,
            alerts: []
        };
        
        this.colors = {
            reset: '\x1b[0m',
            red: '\x1b[31m',
            green: '\x1b[32m',
            yellow: '\x1b[33m',
            blue: '\x1b[34m',
            cyan: '\x1b[36m'
        };
    }

    log(level, message, data = null) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            message,
            data
        };
        
        // Console output with colors
        const color = {
            'INFO': this.colors.blue,
            'SUCCESS': this.colors.green,
            'WARNING': this.colors.yellow,
            'ERROR': this.colors.red,
            'CRITICAL': this.colors.red
        }[level] || this.colors.reset;
        
        console.log(`${color}[${timestamp}] ${level}: ${message}${this.colors.reset}`);
        if (data) {
            console.log('Data:', JSON.stringify(data, null, 2));
        }
        
        // File logging
        this.writeToHealthLog(logEntry).catch(console.error);
    }

    async writeToHealthLog(entry) {
        try {
            await fs.mkdir(path.dirname(this.config.healthLogFile), { recursive: true });
            const logLine = JSON.stringify(entry) + '\n';
            await fs.appendFile(this.config.healthLogFile, logLine);
        } catch (error) {
            console.error('Failed to write health log:', error);
        }
    }

    async getMCPPid() {
        try {
            const pidContent = await fs.readFile(this.config.pidFile, 'utf-8');
            return parseInt(pidContent.trim());
        } catch (error) {
            return null;
        }
    }

    async isProcessRunning(pid) {
        try {
            process.kill(pid, 0);
            return true;
        } catch (error) {
            return false;
        }
    }

    async getProcessStats(pid) {
        return new Promise((resolve, reject) => {
            const ps = spawn('ps', ['-p', pid.toString(), '-o', 'pid,ppid,%cpu,%mem,rss,vsz,time,cmd', '--no-headers']);
            let output = '';
            let error = '';
            
            ps.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            ps.stderr.on('data', (data) => {
                error += data.toString();
            });
            
            ps.on('close', (code) => {
                if (code === 0 && output.trim()) {
                    const parts = output.trim().split(/\s+/);
                    if (parts.length >= 8) {
                        resolve({
                            pid: parseInt(parts[0]),
                            ppid: parseInt(parts[1]),
                            cpuPercent: parseFloat(parts[2]),
                            memPercent: parseFloat(parts[3]),
                            memoryMB: Math.round(parseInt(parts[4]) / 1024),
                            virtualMemoryMB: Math.round(parseInt(parts[5]) / 1024),
                            time: parts[6],
                            command: parts.slice(7).join(' ')
                        });
                    } else {
                        reject(new Error('Invalid ps output format'));
                    }
                } else {
                    reject(new Error(error || 'Process not found'));
                }
            });
        });
    }

    async testMCPConnection() {
        const startTime = performance.now();
        
        return new Promise((resolve) => {
            const timeout = setTimeout(() => {
                resolve({
                    success: false,
                    error: 'Connection timeout',
                    responseTime: performance.now() - startTime
                });
            }, this.config.responseTimeoutMs);
            
            try {
                // Test basic MCP functionality by spawning a quick test
                const testProcess = spawn('npx', ['claude-flow@alpha', 'memory', 'list', '--namespace', 'health-check'], {
                    timeout: this.config.responseTimeoutMs,
                    stdio: ['pipe', 'pipe', 'pipe']
                });
                
                let output = '';
                let errorOutput = '';
                
                testProcess.stdout?.on('data', (data) => {
                    output += data.toString();
                });
                
                testProcess.stderr?.on('data', (data) => {
                    errorOutput += data.toString();
                });
                
                testProcess.on('close', (code) => {
                    clearTimeout(timeout);
                    const responseTime = performance.now() - startTime;
                    
                    if (code === 0 || output.includes('success') || output.includes('entries')) {
                        resolve({
                            success: true,
                            responseTime,
                            output: output.trim()
                        });
                    } else {
                        resolve({
                            success: false,
                            error: errorOutput || 'Command failed',
                            responseTime,
                            code
                        });
                    }
                });
                
                testProcess.on('error', (error) => {
                    clearTimeout(timeout);
                    resolve({
                        success: false,
                        error: error.message,
                        responseTime: performance.now() - startTime
                    });
                });
                
            } catch (error) {
                clearTimeout(timeout);
                resolve({
                    success: false,
                    error: error.message,
                    responseTime: performance.now() - startTime
                });
            }
        });
    }

    checkThresholds(stats, responseTime) {
        const alerts = [];
        const { memory, cpu, responseTime: respTime } = this.config.alertThresholds;
        
        // Memory checks
        if (stats.memoryMB >= memory.critical) {
            alerts.push({ type: 'CRITICAL', metric: 'memory', value: stats.memoryMB, threshold: memory.critical });
        } else if (stats.memoryMB >= memory.warning) {
            alerts.push({ type: 'WARNING', metric: 'memory', value: stats.memoryMB, threshold: memory.warning });
        }
        
        // CPU checks
        if (stats.cpuPercent >= cpu.critical) {
            alerts.push({ type: 'CRITICAL', metric: 'cpu', value: stats.cpuPercent, threshold: cpu.critical });
        } else if (stats.cpuPercent >= cpu.warning) {
            alerts.push({ type: 'WARNING', metric: 'cpu', value: stats.cpuPercent, threshold: cpu.warning });
        }
        
        // Response time checks
        if (responseTime >= respTime.critical) {
            alerts.push({ type: 'CRITICAL', metric: 'response_time', value: responseTime, threshold: respTime.critical });
        } else if (responseTime >= respTime.warning) {
            alerts.push({ type: 'WARNING', metric: 'response_time', value: responseTime, threshold: respTime.warning });
        }
        
        return alerts;
    }

    async generateHealthReport() {
        try {
            // Get current health status
            const healthStatus = await this.performHealthCheck();
            
            // Calculate uptime
            const uptime = Date.now() - this.metrics.startTime;
            const uptimeHours = Math.floor(uptime / (1000 * 60 * 60));
            const uptimeMinutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
            
            // Success rate
            const successRate = this.metrics.checks > 0 ? 
                ((this.metrics.checks - this.metrics.failures) / this.metrics.checks * 100).toFixed(2) : 0;
            
            const report = {
                timestamp: new Date().toISOString(),
                uptime: `${uptimeHours}h ${uptimeMinutes}m`,
                healthStatus: healthStatus.status,
                metrics: {
                    totalChecks: this.metrics.checks,
                    failures: this.metrics.failures,
                    successRate: `${successRate}%`,
                    averageResponseTime: `${this.metrics.averageResponseTime.toFixed(2)}ms`
                },
                currentStatus: healthStatus,
                recentAlerts: this.metrics.alerts.slice(-10)
            };
            
            // Save report
            const reportPath = path.join(this.config.logDir, 'health-report.json');
            await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
            
            return report;
        } catch (error) {
            this.log('ERROR', 'Failed to generate health report', { error: error.message });
            throw error;
        }
    }

    async performHealthCheck() {
        this.metrics.checks++;
        this.metrics.lastCheck = Date.now();
        
        try {
            this.log('INFO', 'Starting health check...');
            
            // 1. Check if MCP server process is running
            const pid = await this.getMCPPid();
            if (!pid) {
                this.metrics.failures++;
                this.log('ERROR', 'MCP server PID file not found');
                return { status: 'ERROR', error: 'PID file not found' };
            }
            
            const isRunning = await this.isProcessRunning(pid);
            if (!isRunning) {
                this.metrics.failures++;
                this.log('ERROR', 'MCP server process not running', { pid });
                return { status: 'ERROR', error: 'Process not running', pid };
            }
            
            // 2. Get process statistics
            const stats = await getProcessStats(pid);
            this.log('INFO', 'Process statistics', stats);
            
            // 3. Test MCP connection and functionality
            const connectionTest = await this.testMCPConnection();
            
            // Update average response time
            if (this.metrics.averageResponseTime === 0) {
                this.metrics.averageResponseTime = connectionTest.responseTime;
            } else {
                this.metrics.averageResponseTime = 
                    (this.metrics.averageResponseTime + connectionTest.responseTime) / 2;
            }
            
            if (!connectionTest.success) {
                this.metrics.failures++;
                this.log('ERROR', 'MCP connection test failed', connectionTest);
                return {
                    status: 'ERROR',
                    error: 'Connection test failed',
                    details: connectionTest,
                    processStats: stats
                };
            }
            
            // 4. Check thresholds and generate alerts
            const alerts = this.checkThresholds(stats, connectionTest.responseTime);
            
            if (alerts.length > 0) {
                this.metrics.alerts.push(...alerts.map(alert => ({
                    ...alert,
                    timestamp: Date.now()
                })));
                
                const criticalAlerts = alerts.filter(a => a.type === 'CRITICAL');
                if (criticalAlerts.length > 0) {
                    criticalAlerts.forEach(alert => {
                        this.log('CRITICAL', `Critical threshold exceeded: ${alert.metric}`, alert);
                    });
                    return {
                        status: 'CRITICAL',
                        processStats: stats,
                        connectionTest,
                        alerts
                    };
                } else {
                    alerts.forEach(alert => {
                        this.log('WARNING', `Warning threshold exceeded: ${alert.metric}`, alert);
                    });
                }
            }
            
            // 5. All checks passed
            this.log('SUCCESS', 'Health check completed successfully', {
                pid,
                memory: `${stats.memoryMB}MB`,
                cpu: `${stats.cpuPercent}%`,
                responseTime: `${connectionTest.responseTime.toFixed(2)}ms`
            });
            
            return {
                status: 'HEALTHY',
                pid,
                processStats: stats,
                connectionTest,
                alerts: alerts.length > 0 ? alerts : null
            };
            
        } catch (error) {
            this.metrics.failures++;
            this.log('ERROR', 'Health check failed', { error: error.message });
            return { status: 'ERROR', error: error.message };
        }
    }

    async startContinuousMonitoring() {
        this.log('INFO', `Starting continuous health monitoring (interval: ${this.config.healthInterval}ms)`);
        
        const monitor = async () => {
            try {
                await this.performHealthCheck();
            } catch (error) {
                this.log('ERROR', 'Monitoring cycle failed', { error: error.message });
            }
        };
        
        // Initial check
        await monitor();
        
        // Set up recurring checks
        const interval = setInterval(monitor, this.config.healthInterval);
        
        // Graceful shutdown
        process.on('SIGINT', () => {
            this.log('INFO', 'Stopping health monitor...');
            clearInterval(interval);
            process.exit(0);
        });
        
        process.on('SIGTERM', () => {
            this.log('INFO', 'Stopping health monitor...');
            clearInterval(interval);
            process.exit(0);
        });
    }
}

// Helper function to get process stats (needs to be accessible)
async function getProcessStats(pid) {
    return new Promise((resolve, reject) => {
        const ps = spawn('ps', ['-p', pid.toString(), '-o', 'pid,ppid,%cpu,%mem,rss,vsz,time,cmd', '--no-headers']);
        let output = '';
        let error = '';
        
        ps.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        ps.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        ps.on('close', (code) => {
            if (code === 0 && output.trim()) {
                const parts = output.trim().split(/\s+/);
                if (parts.length >= 8) {
                    resolve({
                        pid: parseInt(parts[0]),
                        ppid: parseInt(parts[1]),
                        cpuPercent: parseFloat(parts[2]),
                        memPercent: parseFloat(parts[3]),
                        memoryMB: Math.round(parseInt(parts[4]) / 1024),
                        virtualMemoryMB: Math.round(parseInt(parts[5]) / 1024),
                        time: parts[6],
                        command: parts.slice(7).join(' ')
                    });
                } else {
                    reject(new Error('Invalid ps output format'));
                }
            } else {
                reject(new Error(error || 'Process not found'));
            }
        });
    });
}

// CLI interface
async function main() {
    const monitor = new MCPHealthMonitor();
    const command = process.argv[2] || 'check';
    
    switch (command) {
        case 'check':
            const result = await monitor.performHealthCheck();
            console.log('\n' + JSON.stringify(result, null, 2));
            process.exit(result.status === 'HEALTHY' ? 0 : 1);
            break;
            
        case 'monitor':
            await monitor.startContinuousMonitoring();
            break;
            
        case 'report':
            const report = await monitor.generateHealthReport();
            console.log('\n' + JSON.stringify(report, null, 2));
            break;
            
        default:
            console.log('Usage: node health-check.js [check|monitor|report]');
            console.log('  check   - Run single health check');
            console.log('  monitor - Start continuous monitoring');
            console.log('  report  - Generate health report');
            process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(error => {
        console.error('Health check failed:', error);
        process.exit(1);
    });
}

module.exports = { MCPHealthMonitor };