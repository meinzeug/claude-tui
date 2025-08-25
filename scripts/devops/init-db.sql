-- Claude-TIU Database Initialization Script
-- Creates necessary databases, users, and initial schema

-- Create database (if not exists)
SELECT 'CREATE DATABASE claude_tui' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'claude_tui')\gexec

-- Connect to the claude_tui database
\c claude_tui;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}'::jsonb
);

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    config JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- Create AI sessions table
CREATE TABLE IF NOT EXISTS ai_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    session_type VARCHAR(50) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) DEFAULT 'active',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    metrics JSONB DEFAULT '{}'::jsonb
);

-- Create AI tasks table
CREATE TABLE IF NOT EXISTS ai_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES ai_sessions(id) ON DELETE CASCADE,
    task_type VARCHAR(50) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    tokens_used INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create validation results table
CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES ai_tasks(id) ON DELETE CASCADE,
    validation_type VARCHAR(50) NOT NULL,
    result BOOLEAN NOT NULL,
    score DECIMAL(3,2),
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create metrics tables
CREATE TABLE IF NOT EXISTS metrics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    unit VARCHAR(20),
    tags JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS metrics.system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_rate DECIMAL(5,4),
    details JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create file storage table (for metadata)
CREATE TABLE IF NOT EXISTS file_storage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    checksum TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    key_hash TEXT NOT NULL,
    permissions JSONB DEFAULT '[]'::jsonb,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_ai_sessions_project_id ON ai_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_ai_sessions_status ON ai_sessions(status);
CREATE INDEX IF NOT EXISTS idx_ai_tasks_session_id ON ai_tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_ai_tasks_status ON ai_tasks(status);
CREATE INDEX IF NOT EXISTS idx_ai_tasks_created ON ai_tasks(started_at);
CREATE INDEX IF NOT EXISTS idx_validation_results_task_id ON validation_results(task_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit.audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit.audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON metrics.performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON metrics.performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_component ON metrics.system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON metrics.system_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_file_storage_project_id ON file_storage(project_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_file_storage_updated_at BEFORE UPDATE ON file_storage FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for audit logging
CREATE OR REPLACE FUNCTION audit.log_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit.audit_log (
        action,
        resource_type,
        resource_id,
        changes
    ) VALUES (
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE
            WHEN TG_OP = 'DELETE' THEN row_to_json(OLD)
            WHEN TG_OP = 'INSERT' THEN row_to_json(NEW)
            ELSE jsonb_build_object(
                'old', row_to_json(OLD),
                'new', row_to_json(NEW)
            )
        END
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers
CREATE TRIGGER users_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

CREATE TRIGGER projects_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON projects
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- Create a view for project statistics
CREATE OR REPLACE VIEW project_stats AS
SELECT 
    p.id,
    p.name,
    p.created_at,
    COUNT(DISTINCT s.id) as session_count,
    COUNT(DISTINCT t.id) as task_count,
    AVG(t.execution_time_ms) as avg_execution_time_ms,
    SUM(t.tokens_used) as total_tokens_used,
    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN t.status = 'failed' THEN 1 END) as failed_tasks
FROM projects p
LEFT JOIN ai_sessions s ON p.id = s.project_id
LEFT JOIN ai_tasks t ON s.id = t.session_id
GROUP BY p.id, p.name, p.created_at;

-- Create a view for system health dashboard
CREATE OR REPLACE VIEW system_health_dashboard AS
SELECT 
    component,
    status,
    AVG(response_time_ms) as avg_response_time_ms,
    AVG(error_rate) as avg_error_rate,
    MAX(timestamp) as last_check
FROM metrics.system_health
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY component, status
ORDER BY component;

-- Insert default admin user (change password in production!)
INSERT INTO users (username, email, password_hash) 
VALUES ('admin', 'admin@claude-tui.local', crypt('admin123', gen_salt('bf')))
ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA audit TO postgres;
GRANT USAGE ON SCHEMA metrics TO postgres;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO postgres;
GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO postgres;

-- Create database health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE (
    component TEXT,
    status TEXT,
    details JSONB
) AS $$
BEGIN
    -- Check database connectivity
    RETURN QUERY SELECT 
        'database'::TEXT,
        'healthy'::TEXT,
        jsonb_build_object(
            'timestamp', NOW(),
            'version', version(),
            'active_connections', (SELECT count(*) FROM pg_stat_activity)
        );
    
    -- Check table row counts
    RETURN QUERY SELECT 
        'data_integrity'::TEXT,
        'healthy'::TEXT,
        jsonb_build_object(
            'users_count', (SELECT count(*) FROM users),
            'projects_count', (SELECT count(*) FROM projects),
            'tasks_count', (SELECT count(*) FROM ai_tasks)
        );
END;
$$ LANGUAGE plpgsql;

-- Create cleanup function for old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    rows_deleted INTEGER := 0;
BEGIN
    -- Clean old audit logs
    DELETE FROM audit.audit_log WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS rows_deleted = ROW_COUNT;
    
    -- Clean old performance metrics (keep longer retention)
    DELETE FROM metrics.performance_metrics WHERE timestamp < NOW() - INTERVAL '1 day' * (days_to_keep * 3);
    
    -- Clean old system health records
    DELETE FROM metrics.system_health WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    
    RETURN rows_deleted;
END;
$$ LANGUAGE plpgsql;

-- Create user for application connection
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'claude_app') THEN
        CREATE USER claude_app WITH ENCRYPTED PASSWORD 'claude_secure_app_password';
    END IF;
END
$$;

-- Grant necessary permissions to application user
GRANT CONNECT ON DATABASE claude_tui TO claude_app;
GRANT USAGE ON SCHEMA public TO claude_app;
GRANT USAGE ON SCHEMA audit TO claude_app;
GRANT USAGE ON SCHEMA metrics TO claude_app;
GRANT ALL ON ALL TABLES IN SCHEMA public TO claude_app;
GRANT INSERT ON audit.audit_log TO claude_app;
GRANT ALL ON metrics.performance_metrics TO claude_app;
GRANT ALL ON metrics.system_health TO claude_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO claude_app;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO claude_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO claude_app;

-- Create a scheduled job for cleanup (if pg_cron extension is available)
-- SELECT cron.schedule('cleanup-claude-tui', '0 2 * * *', 'SELECT cleanup_old_data(30);');

COMMIT;

-- Display initialization status
SELECT 'Database initialization completed successfully' as status;