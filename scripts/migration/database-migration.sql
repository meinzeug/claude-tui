-- Claude-TIU Production Database Migration Script
-- Version: 1.0.0
-- Description: Initial schema setup for production deployment

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS claude_tiu;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO claude_tiu, public;

-- Users table for authentication and session management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- API keys for service authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE
);

-- Task execution history
CREATE TABLE IF NOT EXISTS task_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    task_type VARCHAR(100) NOT NULL,
    task_description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_percent DECIMAL(5,2),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Swarm coordination state
CREATE TABLE IF NOT EXISTS swarm_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    swarm_id VARCHAR(255) UNIQUE NOT NULL,
    topology VARCHAR(50) NOT NULL,
    agent_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'initializing',
    configuration JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent instances within swarms
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    swarm_id UUID REFERENCES swarm_states(id) ON DELETE CASCADE,
    agent_type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'spawning',
    capabilities JSONB,
    performance_metrics JSONB,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance monitoring data
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'system', 'swarm', 'agent', 'task'
    entity_id UUID,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,4),
    metric_unit VARCHAR(50),
    metadata JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System configuration and settings
CREATE TABLE IF NOT EXISTS configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for security and compliance
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

CREATE INDEX IF NOT EXISTS idx_task_executions_user_id ON task_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_task_type ON task_executions(task_type);
CREATE INDEX IF NOT EXISTS idx_task_executions_started_at ON task_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_swarm_states_swarm_id ON swarm_states(swarm_id);
CREATE INDEX IF NOT EXISTS idx_swarm_states_status ON swarm_states(status);

CREATE INDEX IF NOT EXISTS idx_agents_swarm_id ON agents(swarm_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_entity_type ON performance_metrics(entity_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_entity_id ON performance_metrics(entity_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_type ON performance_metrics(metric_type);

CREATE INDEX IF NOT EXISTS idx_configurations_config_key ON configurations(config_key);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_swarm_states_updated_at BEFORE UPDATE ON swarm_states
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default configuration values
INSERT INTO configurations (config_key, config_value, description) VALUES
    ('max_concurrent_tasks', '100', 'Maximum number of concurrent tasks per user'),
    ('default_task_timeout_seconds', '300', 'Default timeout for task execution'),
    ('max_memory_per_task_mb', '400', 'Maximum memory allocation per task'),
    ('max_cpu_per_task_percent', '70', 'Maximum CPU usage per task'),
    ('performance_monitoring_interval_seconds', '30', 'Interval for performance metric collection'),
    ('swarm_heartbeat_interval_seconds', '10', 'Agent heartbeat interval'),
    ('max_swarm_size', '20', 'Maximum number of agents per swarm'),
    ('enable_audit_logging', 'true', 'Enable comprehensive audit logging'),
    ('session_timeout_minutes', '60', 'User session timeout'),
    ('rate_limit_requests_per_minute', '100', 'API rate limit per user')
ON CONFLICT (config_key) DO NOTHING;

-- Create monitoring schema tables
CREATE TABLE IF NOT EXISTS monitoring.health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_health_checks_service_name ON monitoring.health_checks(service_name);
CREATE INDEX IF NOT EXISTS idx_health_checks_checked_at ON monitoring.health_checks(checked_at);

-- Create views for monitoring
CREATE OR REPLACE VIEW monitoring.active_tasks AS
SELECT 
    te.id,
    te.task_type,
    te.status,
    te.started_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - te.started_at))::INTEGER as runtime_seconds,
    te.memory_usage_mb,
    te.cpu_usage_percent,
    u.username
FROM task_executions te
LEFT JOIN users u ON te.user_id = u.id
WHERE te.status IN ('running', 'pending')
ORDER BY te.started_at DESC;

CREATE OR REPLACE VIEW monitoring.swarm_overview AS
SELECT 
    ss.swarm_id,
    ss.topology,
    ss.status as swarm_status,
    ss.agent_count,
    COUNT(a.id) as active_agents,
    AVG((a.performance_metrics->>'cpu_usage')::DECIMAL) as avg_cpu_usage,
    AVG((a.performance_metrics->>'memory_usage')::DECIMAL) as avg_memory_usage,
    ss.created_at
FROM swarm_states ss
LEFT JOIN agents a ON ss.id = a.swarm_id AND a.status = 'active'
GROUP BY ss.id, ss.swarm_id, ss.topology, ss.status, ss.agent_count, ss.created_at
ORDER BY ss.created_at DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA claude_tiu TO claude_tiu;
GRANT USAGE ON SCHEMA monitoring TO claude_tiu;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA claude_tiu TO claude_tiu;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO claude_tiu;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA claude_tiu TO claude_tiu;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO claude_tiu;

-- Enable row level security for multi-tenant support (optional)
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE task_executions ENABLE ROW LEVEL SECURITY;

COMMIT;