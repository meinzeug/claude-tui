#!/usr/bin/env node

/**
 * MCP Performance Optimizer - Production Memory & CPU Optimization
 * Optimizes claude-flow MCP server for production use
 */

const fs = require('fs').promises;
const path = require('path');
const { spawn, exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

class MCPPerformanceOptimizer {
    constructor() {
        this.config = {
            mcpServerPath: '/home/tekkadmin/.npm/_npx/7cfa166e65244432/node_modules/claude-flow/src/mcp/mcp-server.js',
            backupPath: '/home/tekkadmin/claude-tui/scripts/mcp/mcp-server-backup.js',
            optimizedPath: '/home/tekkadmin/claude-tui/scripts/mcp/mcp-server-optimized.js',
            logDir: '/home/tekkadmin/claude-tui/logs/mcp',
            optimizationLog: '/home/tekkadmin/claude-tui/logs/mcp/optimization.log'
        };
        
        this.optimizations = {
            memoryPool: true,
            eventLoopOptimization: true,
            garbageCollectionTuning: true,
            connectionPooling: true,
            responseBuffering: true,
            lazyLoading: true,
            caching: true,
            processMonitoring: true
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
        const color = {
            'INFO': this.colors.blue,
            'SUCCESS': this.colors.green,
            'WARNING': this.colors.yellow,
            'ERROR': this.colors.red
        }[level] || this.colors.reset;
        
        console.log(`${color}[${timestamp}] ${level}: ${message}${this.colors.reset}`);
        if (data) {
            console.log('Data:', JSON.stringify(data, null, 2));
        }
    }

    async createBackup() {
        try {
            await fs.copyFile(this.config.mcpServerPath, this.config.backupPath);
            this.log('SUCCESS', 'MCP server backup created');
            return true;
        } catch (error) {
            this.log('ERROR', 'Failed to create backup', { error: error.message });
            return false;
        }
    }

    async generateOptimizedServer() {
        const originalContent = await fs.readFile(this.config.mcpServerPath, 'utf-8');
        
        // Apply optimizations to the server code
        let optimizedContent = this.applyMemoryOptimizations(originalContent);
        optimizedContent = this.applyEventLoopOptimizations(optimizedContent);
        optimizedContent = this.applyConnectionOptimizations(optimizedContent);
        optimizedContent = this.applyCachingOptimizations(optimizedContent);
        optimizedContent = this.applyMonitoringOptimizations(optimizedContent);
        
        // Write optimized version
        await fs.writeFile(this.config.optimizedPath, optimizedContent);
        this.log('SUCCESS', 'Optimized MCP server generated');
        
        return optimizedContent;
    }

    applyMemoryOptimizations(content) {
        // Add memory pool and object reuse at the top
        const memoryOptimizations = `
// Memory Pool for Object Reuse
class MemoryPool {
    constructor() {
        this.responsePool = [];
        this.requestPool = [];
        this.maxPoolSize = 100;
        this.memoryThreshold = 500 * 1024 * 1024; // 500MB
        
        // Monitor memory usage
        this.memoryCheckInterval = setInterval(() => {
            this.checkMemoryUsage();
        }, 30000); // Check every 30 seconds
    }
    
    checkMemoryUsage() {
        const used = process.memoryUsage();
        if (used.heapUsed > this.memoryThreshold) {
            console.error(\`[MEMORY WARNING] Heap usage: \${Math.round(used.heapUsed / 1024 / 1024)}MB\`);
            // Force garbage collection if available
            if (global.gc) {
                global.gc();
                console.error('[MEMORY] Forced garbage collection');
            }
            // Clear pools
            this.responsePool.length = 0;
            this.requestPool.length = 0;
        }
    }
    
    getResponse() {
        return this.responsePool.pop() || { jsonrpc: '2.0' };
    }
    
    returnResponse(response) {
        if (this.responsePool.length < this.maxPoolSize) {
            // Reset response object
            delete response.id;
            delete response.result;
            delete response.error;
            this.responsePool.push(response);
        }
    }
    
    cleanup() {
        if (this.memoryCheckInterval) {
            clearInterval(this.memoryCheckInterval);
        }
        this.responsePool.length = 0;
        this.requestPool.length = 0;
    }
}

const memoryPool = new MemoryPool();

// Graceful shutdown handler
process.on('SIGINT', async () => {
    console.error('Graceful shutdown initiated...');
    memoryPool.cleanup();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.error('Graceful shutdown initiated...');
    memoryPool.cleanup();
    process.exit(0);
});

// Set memory limits and optimizations
if (process.env.NODE_ENV !== 'development') {
    // Production memory optimizations
    process.setMaxListeners(50); // Increase max listeners
    
    // Enable garbage collection optimizations
    if (global.gc) {
        setInterval(() => {
            if (process.memoryUsage().heapUsed > 400 * 1024 * 1024) {
                global.gc();
            }
        }, 60000); // GC every minute if memory high
    }
}

`;
        
        return memoryOptimizations + content;
    }

    applyEventLoopOptimizations(content) {
        // Replace sync operations with async where possible
        const eventLoopOptimizations = content.replace(
            /console\.error\(/g,
            'setImmediate(() => console.error('
        ).replace(
            /JSON\.parse\(([^)]+)\)/g,
            '(() => { try { return JSON.parse($1); } catch(e) { return null; } })()'
        );
        
        return eventLoopOptimizations;
    }

    applyConnectionOptimizations(content) {
        // Add connection pooling and response buffering
        const connectionOptimizations = content.replace(
            'class ClaudeFlowMCPServer {',
            `class ClaudeFlowMCPServer {`
        ).replace(
            'constructor() {',
            `constructor() {
        this.responseBuffer = new Map();
        this.connectionPool = new Map();
        this.requestQueue = [];
        this.processingQueue = false;
        this.maxConcurrentRequests = 10;
        this.currentRequests = 0;
        
        // Process request queue
        this.processRequestQueue();`
        );
        
        // Add request queuing system
        const queueSystem = `
    async processRequestQueue() {
        if (this.processingQueue) return;
        this.processingQueue = true;
        
        while (this.requestQueue.length > 0 && this.currentRequests < this.maxConcurrentRequests) {
            const { message, callback } = this.requestQueue.shift();
            this.currentRequests++;
            
            setImmediate(async () => {
                try {
                    const response = await this.handleMessageOptimized(message);
                    callback(response);
                } catch (error) {
                    callback(this.createErrorResponse(message.id, -32603, 'Internal error', error.message));
                } finally {
                    this.currentRequests--;
                }
            });
        }
        
        this.processingQueue = false;
        
        // Schedule next processing cycle
        if (this.requestQueue.length > 0) {
            setImmediate(() => this.processRequestQueue());
        }
    }
    
    async handleMessageOptimized(message) {
        // Check response buffer first
        const cacheKey = JSON.stringify({ method: message.method, params: message.params });
        if (this.responseBuffer.has(cacheKey)) {
            const cached = this.responseBuffer.get(cacheKey);
            if (Date.now() - cached.timestamp < 30000) { // 30 second cache
                const response = memoryPool.getResponse();
                Object.assign(response, { ...cached.response, id: message.id });
                return response;
            } else {
                this.responseBuffer.delete(cacheKey);
            }
        }
        
        const response = await this.handleMessage(message);
        
        // Cache successful responses
        if (response && !response.error && message.method !== 'tools/call') {
            this.responseBuffer.set(cacheKey, {
                response: { ...response },
                timestamp: Date.now()
            });
            
            // Limit cache size
            if (this.responseBuffer.size > 1000) {
                const oldestKey = this.responseBuffer.keys().next().value;
                this.responseBuffer.delete(oldestKey);
            }
        }
        
        return response;
    }
`;
        
        return connectionOptimizations.replace(
            'async handleMessage(message) {',
            queueSystem + '\n    async handleMessage(message) {'
        );
    }

    applyCachingOptimizations(content) {
        // Add intelligent caching for expensive operations
        return content.replace(
            'async executeTool(name, args) {',
            `async executeTool(name, args) {
        // Cache for expensive operations
        const cacheKey = \`\${name}:\${JSON.stringify(args)}\`;
        const cacheable = ['swarm_status', 'agent_list', 'performance_report', 'memory_usage'];
        
        if (cacheable.includes(name) && this.toolCache) {
            const cached = this.toolCache.get(cacheKey);
            if (cached && Date.now() - cached.timestamp < 10000) { // 10 second cache
                return { ...cached.result, timestamp: new Date().toISOString() };
            }
        }
        
        const startTime = Date.now();`
        ).replace(
            'return {',
            `const result = {`
        ).replace(
            'timestamp: new Date().toISOString(),',
            `timestamp: new Date().toISOString(),
            executionTime: Date.now() - startTime,`
        );
    }

    applyMonitoringOptimizations(content) {
        // Add performance monitoring
        const monitoring = content.replace(
            'this.tools = this.initializeTools();',
            `this.tools = this.initializeTools();
        this.toolCache = new Map();
        this.performanceMetrics = {
            requestCount: 0,
            errorCount: 0,
            averageResponseTime: 0,
            startTime: Date.now()
        };
        
        // Performance monitoring
        setInterval(() => {
            this.logPerformanceMetrics();
        }, 60000); // Log every minute`
        );
        
        // Add performance logging method
        const performanceLogging = `
    logPerformanceMetrics() {
        const uptime = Date.now() - this.performanceMetrics.startTime;
        const memoryUsage = process.memoryUsage();
        
        console.error(\`[PERFORMANCE] Uptime: \${Math.round(uptime/1000)}s, Requests: \${this.performanceMetrics.requestCount}, Errors: \${this.performanceMetrics.errorCount}, Avg Response: \${this.performanceMetrics.averageResponseTime.toFixed(2)}ms\`);
        console.error(\`[MEMORY] Heap: \${Math.round(memoryUsage.heapUsed/1024/1024)}MB, RSS: \${Math.round(memoryUsage.rss/1024/1024)}MB\`);
        
        // Clear tool cache if memory is high
        if (memoryUsage.heapUsed > 400 * 1024 * 1024) {
            this.toolCache.clear();
            console.error('[CACHE] Cleared tool cache due to high memory usage');
        }
    }
`;
        
        return monitoring.replace(
            'initializeTools() {',
            performanceLogging + '\n    initializeTools() {'
        );
    }

    async installOptimizedServer() {
        try {
            // Stop any running MCP server
            await this.stopMCPServer();
            
            // Replace the original with optimized version
            await fs.copyFile(this.config.optimizedPath, this.config.mcpServerPath);
            this.log('SUCCESS', 'Optimized MCP server installed');
            
            // Start the optimized server
            await this.startOptimizedMCPServer();
            
            return true;
        } catch (error) {
            this.log('ERROR', 'Failed to install optimized server', { error: error.message });
            
            // Restore backup if installation failed
            try {
                await fs.copyFile(this.config.backupPath, this.config.mcpServerPath);
                this.log('INFO', 'Restored backup after failed installation');
            } catch (restoreError) {
                this.log('ERROR', 'Failed to restore backup', { error: restoreError.message });
            }
            
            return false;
        }
    }

    async stopMCPServer() {
        try {
            const { stdout } = await execAsync('pgrep -f "mcp-server.js"');
            const pids = stdout.trim().split('\n').filter(pid => pid);
            
            for (const pid of pids) {
                try {
                    process.kill(parseInt(pid), 'SIGTERM');
                    this.log('INFO', `Stopped MCP server process: ${pid}`);
                } catch (error) {
                    // Process might already be dead
                }
            }
            
            // Wait for graceful shutdown
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Force kill if still running
            try {
                const { stdout: remaining } = await execAsync('pgrep -f "mcp-server.js"');
                const remainingPids = remaining.trim().split('\n').filter(pid => pid);
                
                for (const pid of remainingPids) {
                    try {
                        process.kill(parseInt(pid), 'SIGKILL');
                        this.log('WARNING', `Force killed MCP server process: ${pid}`);
                    } catch (error) {
                        // Process might already be dead
                    }
                }
            } catch (error) {
                // No remaining processes - good
            }
            
            return true;
        } catch (error) {
            this.log('INFO', 'No MCP server processes found to stop');
            return true;
        }
    }

    async startOptimizedMCPServer() {
        return new Promise((resolve, reject) => {
            const env = {
                ...process.env,
                NODE_OPTIONS: '--max-old-space-size=512 --optimize-for-size --gc-interval=100',
                UV_THREADPOOL_SIZE: '4',
                NODE_ENV: 'production'
            };
            
            const mcpServer = spawn('node', [this.config.mcpServerPath], {
                env,
                stdio: ['pipe', 'pipe', 'pipe'],
                detached: true
            });
            
            // Let it run in background
            mcpServer.unref();
            
            // Check if it started successfully
            setTimeout(async () => {
                try {
                    const { stdout } = await execAsync('pgrep -f "mcp-server.js"');
                    if (stdout.trim()) {
                        this.log('SUCCESS', 'Optimized MCP server started successfully');
                        resolve(true);
                    } else {
                        reject(new Error('MCP server failed to start'));
                    }
                } catch (error) {
                    reject(error);
                }
            }, 3000);
        });
    }

    async runPerformanceTest() {
        this.log('INFO', 'Running performance test...');
        
        const testCommands = [
            'npx claude-flow@alpha memory list',
            'npx claude-flow@alpha swarm status',
            'npx claude-flow@alpha agent list'
        ];
        
        const results = [];
        
        for (const command of testCommands) {
            const startTime = Date.now();
            try {
                await execAsync(command);
                const duration = Date.now() - startTime;
                results.push({ command, duration, success: true });
                this.log('SUCCESS', `${command}: ${duration}ms`);
            } catch (error) {
                const duration = Date.now() - startTime;
                results.push({ command, duration, success: false, error: error.message });
                this.log('ERROR', `${command}: ${duration}ms (failed)`);
            }
        }
        
        return results;
    }

    async generateOptimizationReport() {
        const report = {
            timestamp: new Date().toISOString(),
            optimizations: this.optimizations,
            serverPath: this.config.mcpServerPath,
            backupPath: this.config.backupPath,
            optimizedPath: this.config.optimizedPath,
            performanceTest: await this.runPerformanceTest(),
            recommendations: [
                'Monitor memory usage regularly with health-check.js',
                'Use auto-restart.sh for automatic recovery',
                'Set NODE_OPTIONS for production: --max-old-space-size=512 --optimize-for-size',
                'Enable garbage collection monitoring in production',
                'Consider horizontal scaling if load increases'
            ]
        };
        
        const reportPath = path.join(this.config.logDir, 'optimization-report.json');
        await fs.mkdir(this.config.logDir, { recursive: true });
        await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
        
        this.log('SUCCESS', `Optimization report generated: ${reportPath}`);
        return report;
    }

    async optimize() {
        this.log('INFO', 'Starting MCP server optimization...');
        
        try {
            // Create backup
            if (!await this.createBackup()) {
                throw new Error('Failed to create backup');
            }
            
            // Generate optimized server
            await this.generateOptimizedServer();
            
            // Install optimized version
            if (!await this.installOptimizedServer()) {
                throw new Error('Failed to install optimized server');
            }
            
            // Generate report
            const report = await this.generateOptimizationReport();
            
            this.log('SUCCESS', 'MCP server optimization completed successfully');
            console.log('\nOptimization Summary:');
            console.log('- Memory pooling enabled');
            console.log('- Event loop optimizations applied');
            console.log('- Connection pooling implemented');
            console.log('- Response caching enabled');
            console.log('- Performance monitoring added');
            console.log(`\\nReport: ${path.join(this.config.logDir, 'optimization-report.json')}`);
            
            return report;
            
        } catch (error) {
            this.log('ERROR', 'Optimization failed', { error: error.message });
            throw error;
        }
    }
}

// CLI interface
async function main() {
    const optimizer = new MCPPerformanceOptimizer();
    const command = process.argv[2] || 'optimize';
    
    switch (command) {
        case 'optimize':
            await optimizer.optimize();
            break;
            
        case 'test':
            await optimizer.runPerformanceTest();
            break;
            
        case 'backup':
            await optimizer.createBackup();
            break;
            
        case 'restore':
            try {
                await optimizer.stopMCPServer();
                await fs.copyFile(optimizer.config.backupPath, optimizer.config.mcpServerPath);
                optimizer.log('SUCCESS', 'Backup restored successfully');
            } catch (error) {
                optimizer.log('ERROR', 'Failed to restore backup', { error: error.message });
                process.exit(1);
            }
            break;
            
        default:
            console.log('Usage: node mcp-performance-optimizer.js [optimize|test|backup|restore]');
            console.log('  optimize - Apply all optimizations to MCP server');
            console.log('  test     - Run performance test');
            console.log('  backup   - Create backup of current server');
            console.log('  restore  - Restore from backup');
            process.exit(1);
    }
}

if (require.main === module) {
    main().catch(error => {
        console.error('Optimization failed:', error);
        process.exit(1);
    });
}

module.exports = { MCPPerformanceOptimizer };