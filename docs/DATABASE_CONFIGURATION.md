# Production Database Configuration Guide

## Overview

This guide provides comprehensive instructions for configuring and managing the production database infrastructure for Claude-TUI. The system supports PostgreSQL as the primary database with advanced features including connection pooling, read replicas, Redis caching, automated backups, and health monitoring.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │────│   Load Balancer  │────│  Read Replicas  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │              ┌─────────────────┐               │
         └──────────────│ Master Database │───────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │ Redis Cluster   │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │ Backup Storage  │
                        └─────────────────┘
```

## Components

### 1. Primary PostgreSQL Database
- **Engine**: PostgreSQL 14+ with asyncpg driver
- **Connection**: SSL required in production
- **Pool Size**: 25 connections (configurable)
- **Max Overflow**: 15 additional connections
- **Timeout**: 30 seconds

### 2. Read Replicas
- **Purpose**: Horizontal scaling for read operations
- **Load Balancing**: Weighted random selection
- **Health Monitoring**: Automatic failover
- **Replication Lag**: Monitored (max 5 seconds)

### 3. Redis Cluster
- **Purpose**: Caching and session storage
- **Configuration**: 6-node cluster (3 masters, 3 slaves)
- **Failover**: Automatic with Sentinel
- **SSL**: Enabled in production

### 4. Advanced Connection Pool
- **Features**: Health monitoring, optimization
- **Metrics**: Response time, error rates
- **Recovery**: Automatic connection refresh
- **Monitoring**: Real-time statistics

### 5. Backup System
- **Frequency**: Daily automated backups
- **Retention**: 30 days (configurable)
- **Encryption**: AES-256 encryption
- **Storage**: Local + S3 cloud storage
- **Compression**: gzip compression

### 6. Health Monitoring
- **Checks**: Connection, performance, replication
- **Alerting**: Webhook notifications
- **Recovery**: Automated recovery actions
- **Metrics**: Comprehensive performance tracking

## Environment Configuration

### Required Environment Variables

```bash
# === CORE DATABASE CONFIGURATION ===
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://claude_tui_prod:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=require
DB_HOST=localhost
DB_PORT=5432
DB_NAME=claude_tui_prod
DB_USER=claude_tui_prod
DB_PASSWORD=${POSTGRES_PASSWORD}

# === CONNECTION POOL SETTINGS ===
DB_POOL_SIZE=25
DB_MAX_OVERFLOW=15
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_SSL_REQUIRE=true
DB_QUERY_CACHE_SIZE=1000

# === READ REPLICA CONFIGURATION ===
DB_READ_REPLICA_URLS=postgresql+asyncpg://claude_tui_read:${DB_READ_PASSWORD}@${DB_READ_HOST}:${DB_READ_PORT}/${DB_NAME}?sslmode=require
DB_READ_HOST=localhost
DB_READ_PORT=5433
DB_READ_PASSWORD=${POSTGRES_READ_PASSWORD}

# === REDIS CLUSTER CONFIGURATION ===
REDIS_CLUSTER_NODES=redis-node-1:7000,redis-node-2:7001,redis-node-3:7002,redis-node-4:7003,redis-node-5:7004,redis-node-6:7005
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_SSL=true
REDIS_MAX_CONNECTIONS=50

# === BACKUP CONFIGURATION ===
DB_BACKUP_ENABLED=true
DB_BACKUP_RETENTION_DAYS=30
DB_BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
DB_BACKUP_S3_BUCKET=${BACKUP_S3_BUCKET}
DB_BACKUP_S3_ACCESS_KEY=${AWS_ACCESS_KEY_ID}
DB_BACKUP_S3_SECRET_KEY=${AWS_SECRET_ACCESS_KEY}

# === MONITORING CONFIGURATION ===
DB_ENABLE_METRICS=true
DB_SLOW_QUERY_THRESHOLD=1.0
DB_CONNECTION_MONITORING=true
HEALTH_CHECK_INTERVAL=30
```

### Security Environment Variables

```bash
# === SECURITY CONFIGURATION ===
SECRET_KEY=${APP_SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
BCRYPT_ROUNDS=12
PASSWORD_MIN_LENGTH=12

# === SSL/TLS CONFIGURATION ===
SSL_CERT_PATH=/etc/ssl/certs/claude-tui.crt
SSL_KEY_PATH=/etc/ssl/private/claude-tui.key
SSL_CA_PATH=/etc/ssl/certs/ca-certificates.crt
```

## Setup Instructions

### 1. Initial Setup

Run the automated production database setup script:

```bash
cd /home/tekkadmin/claude-tui
python scripts/setup_production_database.py
```

This script will:
- Validate prerequisites
- Initialize database schema
- Set up connection pooling
- Configure read replicas
- Set up Redis cluster
- Initialize backup system
- Start health monitoring
- Run validation suite

### 2. Manual PostgreSQL Setup

If you need to set up PostgreSQL manually:

```bash
# Create production database
sudo -u postgres createdb claude_tui_prod

# Create production user
sudo -u postgres createuser claude_tui_prod

# Set password and permissions
sudo -u postgres psql -c "ALTER USER claude_tui_prod WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE claude_tui_prod TO claude_tui_prod;"

# Enable SSL (edit postgresql.conf)
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
```

### 3. Redis Cluster Setup

Set up Redis cluster for caching:

```bash
# Install Redis
sudo apt-get install redis-server

# Configure cluster nodes (repeat for each node)
# Edit /etc/redis/redis.conf:
port 7000
cluster-enabled yes
cluster-config-file nodes-7000.conf
cluster-node-timeout 5000
appendonly yes
```

### 4. Read Replica Setup

Configure PostgreSQL read replicas:

```bash
# On master server (postgresql.conf)
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'

# On replica server
# Create base backup and configure recovery.conf
```

## Database Schema Management

### Alembic Migrations

The system uses Alembic for database schema management:

```bash
# Generate new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Schema Structure

The database includes the following core tables:

- **users**: User accounts and authentication
- **roles**: RBAC role definitions
- **permissions**: Granular permissions
- **user_roles**: User-role assignments
- **user_sessions**: Session tracking
- **projects**: Project management
- **tasks**: Task tracking
- **audit_logs**: Security audit trail

## Performance Optimization

### Connection Pool Tuning

Optimal connection pool settings based on workload:

```python
# High-traffic production
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=25

# Standard production
DB_POOL_SIZE=25
DB_MAX_OVERFLOW=15

# Development
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=2
```

### Query Optimization

- **Indexes**: Automatically created for foreign keys and frequently queried columns
- **Query Cache**: 1000 queries cached by default
- **Slow Query Monitoring**: Queries >1 second logged
- **Connection Pre-ping**: Validates connections before use

### Read Replica Load Balancing

Available strategies:
- `WEIGHTED_RANDOM`: Performance-based weighting (default)
- `ROUND_ROBIN`: Simple round-robin distribution
- `LEAST_CONNECTIONS`: Route to least loaded replica
- `FASTEST_RESPONSE`: Route to fastest responding replica
- `HEALTH_BASED`: Route based on overall health score

## Backup and Recovery

### Automated Backups

- **Schedule**: Daily at 2 AM (configurable)
- **Retention**: 30 days (configurable)
- **Compression**: gzip compression
- **Encryption**: AES-256 encryption
- **Storage**: Local filesystem + S3

### Manual Backup

Create manual backup:

```bash
# Using the backup system
python -c "
import asyncio
from src.database.backup_recovery import setup_backup_manager, BackupConfig

async def backup():
    config = BackupConfig(enabled=True, local_path='/tmp/manual_backup')
    manager = await setup_backup_manager('postgresql://...', config)
    await manager.create_full_backup('manual_backup_$(date +%Y%m%d)')

asyncio.run(backup())
"
```

### Recovery Process

Restore from backup:

```bash
# Using the recovery system
python -c "
import asyncio
from src.database.backup_recovery import setup_backup_manager, BackupConfig

async def restore():
    config = BackupConfig(enabled=True, local_path='/var/backups/claude-tui')
    manager = await setup_backup_manager('postgresql://...', config)
    await manager.restore_from_backup('backup_id', 'claude_tui_prod_restored')

asyncio.run(restore())
"
```

## Monitoring and Health Checks

### Health Check Endpoints

The system provides comprehensive health monitoring:

```python
# Get health status
from src.database.health_monitor import get_health_monitor

monitor = get_health_monitor()
status = await monitor.get_health_status()
```

### Key Metrics

Monitored metrics include:
- **Connection Health**: Response time, availability
- **Query Performance**: Average execution time
- **Connection Count**: Active vs. available connections
- **Database Size**: Storage usage
- **Replication Lag**: Master-replica synchronization
- **Error Rates**: Query failure rates

### Alerting

Configure webhook alerts:

```bash
# Set webhook URL for alerts
ALERT_WEBHOOK_URL=https://your-monitoring-system.com/webhook
```

## Validation and Testing

### Production Validation

Run comprehensive validation:

```bash
# Run full validation suite
python src/database/production_validator.py

# Save results to file
python src/database/production_validator.py > validation_report.json
```

### Performance Testing

Load test the database:

```python
# Connection pool performance test
async def load_test():
    # Creates 20+ concurrent connections
    tasks = [test_connection() for _ in range(20)]
    await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Issues

#### 1. Connection Pool Exhausted

**Symptoms**: `QueuePool limit exceeded` errors

**Solutions**:
```bash
# Increase pool size
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=25

# Reduce pool timeout
DB_POOL_TIMEOUT=10
```

#### 2. High Replication Lag

**Symptoms**: Read replicas lag behind master

**Solutions**:
- Check network connectivity
- Monitor disk I/O on replica
- Increase `max_wal_senders`
- Optimize replica hardware

#### 3. Redis Connection Issues

**Symptoms**: Cache operations failing

**Solutions**:
```bash
# Check Redis cluster status
redis-cli cluster nodes

# Test individual nodes
redis-cli -h node-1 -p 7000 ping
```

#### 4. Backup Failures

**Symptoms**: Automated backups failing

**Solutions**:
- Check disk space in backup directory
- Verify S3 credentials
- Test manual backup
- Check PostgreSQL user permissions

### Diagnostic Commands

```bash
# Check database connectivity
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Monitor active connections
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Check replication status
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
SELECT client_addr, state, sync_state FROM pg_stat_replication;"

# Redis cluster status
redis-cli --cluster check redis-node-1:7000
```

## Security Best Practices

### Database Security

1. **SSL/TLS**: Always use encrypted connections
2. **Authentication**: Strong passwords, certificate-based auth
3. **Network**: Firewall rules, VPN access
4. **Auditing**: Enable query logging, audit trails
5. **Backup Encryption**: Encrypt backup files

### Access Control

1. **Principle of Least Privilege**: Minimal required permissions
2. **Role-Based Access**: Use database roles
3. **Connection Limits**: Limit concurrent connections
4. **IP Restrictions**: Whitelist allowed IP addresses

### Monitoring

1. **Failed Logins**: Monitor authentication failures
2. **Suspicious Queries**: Alert on unusual query patterns
3. **Privilege Escalation**: Monitor permission changes
4. **Data Access**: Log sensitive data access

## Maintenance

### Regular Tasks

#### Daily
- Monitor backup completion
- Check health status
- Review error logs
- Validate replication lag

#### Weekly
- Review performance metrics
- Clean up old backup files
- Update database statistics
- Check disk space usage

#### Monthly
- Review user access permissions
- Update security patches
- Performance optimization review
- Disaster recovery testing

### Maintenance Windows

Schedule maintenance during low-usage periods:

```bash
# Graceful shutdown for maintenance
# 1. Stop new connections
# 2. Wait for active transactions
# 3. Perform maintenance
# 4. Restart services
```

## High Availability Setup

### Multi-Master Configuration

For maximum availability, consider multi-master setup:

```yaml
# PostgreSQL with Patroni
patroni:
  cluster_name: claude-tui-cluster
  postgresql:
    data_dir: /var/lib/postgresql/data
    bin_dir: /usr/lib/postgresql/14/bin
    
# Redis Sentinel
sentinel:
  - host: sentinel-1
    port: 26379
  - host: sentinel-2
    port: 26379
  - host: sentinel-3
    port: 26379
```

### Load Balancer Configuration

Configure HAProxy or similar:

```
backend postgres_backend
    balance roundrobin
    option httpchk GET /health
    server postgres1 10.0.1.10:5432 check
    server postgres2 10.0.1.11:5432 check backup
```

## API Reference

### Database Manager

```python
from src.database.session import DatabaseManager, DatabaseConfig

# Initialize database
config = DatabaseConfig(database_url="postgresql://...")
manager = DatabaseManager(config)
await manager.initialize()

# Get session
async with manager.get_session() as session:
    # Execute queries
    result = await session.execute(text("SELECT * FROM users"))
```

### Connection Pool Manager

```python
from src.database.connection_pool import setup_advanced_connection_pool

# Setup advanced pool
pool = await setup_advanced_connection_pool(
    engine=engine,
    min_pool_size=10,
    max_pool_size=50,
    auto_optimize=True
)

# Get statistics
stats = await pool.get_pool_statistics()
```

### Backup Manager

```python
from src.database.backup_recovery import setup_backup_manager, BackupConfig

# Configure backups
config = BackupConfig(
    enabled=True,
    retention_days=30,
    s3_bucket="my-backups"
)

# Create backup
manager = await setup_backup_manager(database_url, config)
backup = await manager.create_full_backup()
```

## Support and Resources

### Documentation
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

### Monitoring Tools
- pgAdmin for PostgreSQL management
- Redis Commander for Redis management
- Grafana for metrics visualization
- Prometheus for metrics collection

### Community Resources
- PostgreSQL Community
- Redis Community
- SQLAlchemy Community

---

For additional support or questions about database configuration, please refer to the project documentation or contact the development team.