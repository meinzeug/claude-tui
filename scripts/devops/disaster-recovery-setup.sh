#!/bin/bash
# Disaster Recovery Setup for Claude-TUI
# DevOps Swarm Coordination - Final Implementation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
DR_DIR="${PROJECT_ROOT}/disaster-recovery"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR)
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp] WARN: $message${NC}"
            ;;
        INFO)
            echo -e "${GREEN}[$timestamp] INFO: $message${NC}"
            ;;
        DEBUG)
            echo -e "${BLUE}[$timestamp] DEBUG: $message${NC}"
            ;;
    esac
}

# Setup backup infrastructure
setup_backup_infrastructure() {
    log INFO "Setting up backup infrastructure..."
    
    mkdir -p "${BACKUP_DIR}"/{daily,weekly,monthly,emergency}
    mkdir -p "${DR_DIR}"/{scripts,configs,runbooks}
    
    # Database backup script
    cat > "${DR_DIR}/scripts/database-backup.sh" << 'EOF'
#!/bin/bash
# Automated Database Backup

set -euo pipefail

BACKUP_TYPE=${1:-"daily"}
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
BACKUP_DIR="/backups/${BACKUP_TYPE}"
RETENTION_DAYS=${2:-7}

# Database connection
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-claude_tui}"
DB_USER="${DB_USER:-claude_user}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

create_backup() {
    local backup_file="${BACKUP_DIR}/claude_tui_${TIMESTAMP}.sql.gz"
    
    log "Creating database backup: $backup_file"
    
    PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --clean \
        --create \
        --format=custom | gzip > "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log "✅ Database backup created successfully: $backup_file"
        
        # Create manifest
        cat > "${backup_file}.manifest" << EOF_MANIFEST
{
    "timestamp": "$(date -Iseconds)",
    "type": "$BACKUP_TYPE",
    "database": "$DB_NAME",
    "file": "$backup_file",
    "size": "$(stat -c%s "$backup_file")",
    "checksum": "$(sha256sum "$backup_file" | cut -d' ' -f1)"
}
EOF_MANIFEST
        
        return 0
    else
        log "❌ Database backup failed"
        return 1
    fi
}

cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "*.manifest" -mtime +$RETENTION_DAYS -delete
    
    log "✅ Old backups cleaned up"
}

main() {
    mkdir -p "$BACKUP_DIR"
    
    if create_backup; then
        cleanup_old_backups
        log "🎉 Backup process completed successfully"
    else
        log "💥 Backup process failed"
        exit 1
    fi
}

main "$@"
EOF

    chmod +x "${DR_DIR}/scripts/database-backup.sh"
    
    # Full system backup script
    cat > "${DR_DIR}/scripts/full-system-backup.sh" << 'EOF'
#!/bin/bash
# Full System Backup for Disaster Recovery

set -euo pipefail

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
BACKUP_ROOT="/backups/full-system"
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"
SOURCE_DIRS=(
    "/opt/claude-tui"
    "/etc/systemd/system/claude-tui*"
    "/etc/nginx/sites-available/claude-tui*"
    "/etc/ssl/certs/claude-tui*"
)

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

create_full_backup() {
    log "Creating full system backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup application code and data
    for source_dir in "${SOURCE_DIRS[@]}"; do
        if [[ -e "$source_dir" ]]; then
            log "Backing up: $source_dir"
            tar -czf "$BACKUP_DIR/$(basename "$source_dir").tar.gz" -C "$(dirname "$source_dir")" "$(basename "$source_dir")"
        fi
    done
    
    # Backup Docker volumes
    docker run --rm \
        -v claude-tui_postgres_data:/data/postgres:ro \
        -v claude-tui_redis_data:/data/redis:ro \
        -v "$BACKUP_DIR":/backup \
        alpine:latest \
        sh -c 'tar -czf /backup/docker-volumes.tar.gz -C /data .'
    
    # Create system manifest
    cat > "$BACKUP_DIR/system-manifest.json" << EOF_MANIFEST
{
    "timestamp": "$(date -Iseconds)",
    "type": "full-system",
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "docker_version": "$(docker --version)",
    "kubernetes_version": "$(kubectl version --client --short 2>/dev/null || echo 'N/A')",
    "backup_size": "$(du -sh "$BACKUP_DIR" | cut -f1)"
}
EOF_MANIFEST
    
    log "✅ Full system backup completed: $BACKUP_DIR"
}

sync_to_remote() {
    local remote_path="${REMOTE_BACKUP_PATH:-""}"
    
    if [[ -n "$remote_path" ]]; then
        log "Syncing backup to remote location..."
        rsync -av --progress "$BACKUP_DIR/" "$remote_path/full-system-$TIMESTAMP/"
        log "✅ Backup synced to remote location"
    fi
}

main() {
    create_full_backup
    sync_to_remote
    
    log "🎉 Full system backup completed successfully"
}

main "$@"
EOF

    chmod +x "${DR_DIR}/scripts/full-system-backup.sh"
    
    log INFO "Backup infrastructure setup completed"
}

# Setup disaster recovery automation
setup_dr_automation() {
    log INFO "Setting up disaster recovery automation..."
    
    # Health monitoring and automatic failover
    cat > "${DR_DIR}/scripts/health-monitor-failover.sh" << 'EOF'
#!/bin/bash
# Health Monitoring with Automatic Failover

set -euo pipefail

PRIMARY_URL="${PRIMARY_URL:-https://claude-tui.dev}"
BACKUP_URL="${BACKUP_URL:-https://backup.claude-tui.dev}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
FAILURE_THRESHOLD="${FAILURE_THRESHOLD:-3}"
DNS_UPDATE_SCRIPT="${DNS_UPDATE_SCRIPT:-/opt/claude-tui/scripts/update-dns.sh}"

failure_count=0
current_active="primary"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/claude-tui-failover.log
}

check_health() {
    local url=$1
    local timeout=10
    
    if curl -f -s --max-time $timeout "$url/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

failover_to_backup() {
    log "🚨 INITIATING FAILOVER TO BACKUP SYSTEM"
    
    # Update DNS to point to backup
    if [[ -x "$DNS_UPDATE_SCRIPT" ]]; then
        "$DNS_UPDATE_SCRIPT" "$BACKUP_URL"
        log "✅ DNS updated to point to backup system"
    fi
    
    # Send critical alert
    curl -X POST -H "Content-Type: application/json" \
        -d "{
            \"text\": \"🚨 CRITICAL: Claude-TUI failed over to backup system\",
            \"attachments\": [{
                \"color\": \"danger\",
                \"title\": \"Automatic Failover Triggered\",
                \"text\": \"Primary system health checks failed $failure_count times. Traffic redirected to backup system.\",
                \"fields\": [{
                    \"title\": \"Primary URL\",
                    \"value\": \"$PRIMARY_URL\",
                    \"short\": true
                }, {
                    \"title\": \"Backup URL\",
                    \"value\": \"$BACKUP_URL\",
                    \"short\": true
                }, {
                    \"title\": \"Timestamp\",
                    \"value\": \"$(date -Iseconds)\",
                    \"short\": true
                }]
            }]
        }" \
        "${SLACK_WEBHOOK_URL:-}" || true
    
    current_active="backup"
    failure_count=0
}

failback_to_primary() {
    log "🔄 INITIATING FAILBACK TO PRIMARY SYSTEM"
    
    # Update DNS back to primary
    if [[ -x "$DNS_UPDATE_SCRIPT" ]]; then
        "$DNS_UPDATE_SCRIPT" "$PRIMARY_URL"
        log "✅ DNS updated back to primary system"
    fi
    
    # Send recovery alert
    curl -X POST -H "Content-Type: application/json" \
        -d "{
            \"text\": \"✅ Claude-TUI recovered - failed back to primary system\",
            \"attachments\": [{
                \"color\": \"good\",
                \"title\": \"Automatic Failback Completed\",
                \"text\": \"Primary system is healthy again. Traffic restored to primary system.\",
                \"fields\": [{
                    \"title\": \"Primary URL\",
                    \"value\": \"$PRIMARY_URL\",
                    \"short\": true
                }, {
                    \"title\": \"Timestamp\",
                    \"value\": \"$(date -Iseconds)\",
                    \"short\": true
                }]
            }]
        }" \
        "${SLACK_WEBHOOK_URL:-}" || true
    
    current_active="primary"
}

main() {
    log "Starting health monitoring and failover service..."
    log "Primary: $PRIMARY_URL, Backup: $BACKUP_URL"
    
    while true; do
        if check_health "$PRIMARY_URL"; then
            if [[ "$current_active" == "backup" ]]; then
                # Primary is back up, failback
                failback_to_primary
            fi
            failure_count=0
        else
            ((failure_count++))
            log "⚠️ Primary health check failed ($failure_count/$FAILURE_THRESHOLD)"
            
            if [[ $failure_count -ge $FAILURE_THRESHOLD && "$current_active" == "primary" ]]; then
                if check_health "$BACKUP_URL"; then
                    failover_to_backup
                else
                    log "💥 CRITICAL: Both primary and backup systems are down!"
                fi
            fi
        fi
        
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

main "$@"
EOF

    chmod +x "${DR_DIR}/scripts/health-monitor-failover.sh"
    
    # Create systemd service
    cat > "${DR_DIR}/configs/claude-tui-failover.service" << 'EOF'
[Unit]
Description=Claude-TUI Health Monitor and Failover Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=claude-tui
Group=claude-tui
ExecStart=/opt/claude-tui/disaster-recovery/scripts/health-monitor-failover.sh
Restart=always
RestartSec=10
Environment=PRIMARY_URL=https://claude-tui.dev
Environment=BACKUP_URL=https://backup.claude-tui.dev
Environment=HEALTH_CHECK_INTERVAL=30
Environment=FAILURE_THRESHOLD=3

[Install]
WantedBy=multi-user.target
EOF

    log INFO "Disaster recovery automation setup completed"
}

# Create disaster recovery runbooks
create_dr_runbooks() {
    log INFO "Creating disaster recovery runbooks..."
    
    cat > "${DR_DIR}/runbooks/disaster-recovery-procedures.md" << 'EOF'
# Claude-TUI Disaster Recovery Procedures

## 🚨 Emergency Contact Information

- **Primary On-call**: +1-XXX-XXX-XXXX
- **Secondary On-call**: +1-XXX-XXX-XXXX
- **DevOps Lead**: +1-XXX-XXX-XXXX
- **Slack Channel**: #claude-tui-alerts

## 📋 Recovery Procedures

### 1. Complete System Failure

**Symptoms:**
- Application completely inaccessible
- All health checks failing
- Database connection failures

**Immediate Actions:**
1. Activate incident response team
2. Assess failure scope and impact
3. Initiate failover to backup system
4. Communicate with stakeholders

**Recovery Steps:**
```bash
# 1. Check system status
./disaster-recovery/scripts/health-monitor-failover.sh

# 2. Manual failover if automatic failed
./disaster-recovery/scripts/manual-failover.sh

# 3. Restore from latest backup
./disaster-recovery/scripts/restore-from-backup.sh latest

# 4. Validate system functionality
./scripts/production-validation.sh
```

### 2. Database Corruption/Loss

**Symptoms:**
- Database connection errors
- Data integrity issues
- Transaction failures

**Recovery Steps:**
```bash
# 1. Stop application services
kubectl scale deployment claude-tui-app --replicas=0 -n production

# 2. Restore database from latest backup
./disaster-recovery/scripts/database-restore.sh /backups/daily/latest

# 3. Validate data integrity
./scripts/validate-database.sh

# 4. Restart services
kubectl scale deployment claude-tui-app --replicas=3 -n production
```

### 3. Partial Service Degradation

**Symptoms:**
- High error rates
- Slow response times
- Some features unavailable

**Recovery Steps:**
```bash
# 1. Identify affected components
kubectl get pods -n production
kubectl logs -f deployment/claude-tui-app -n production

# 2. Scale up healthy instances
kubectl scale deployment claude-tui-app --replicas=6 -n production

# 3. Restart unhealthy pods
kubectl delete pods -l app=claude-tui-app -n production

# 4. Monitor recovery
watch kubectl get pods -n production
```

### 4. Security Incident

**Symptoms:**
- Unauthorized access detected
- Unusual traffic patterns
- Security alerts triggered

**Immediate Actions:**
1. **STOP** - Isolate affected systems immediately
2. **ASSESS** - Determine breach scope and impact
3. **CONTAIN** - Prevent further damage
4. **COMMUNICATE** - Notify security team and stakeholders

**Recovery Steps:**
```bash
# 1. Isolate affected systems
kubectl patch networkpolicy deny-all --type=merge -p '{"spec":{"podSelector":{}}}'

# 2. Audit and investigate
./security/scripts/security-audit.sh
./security/scripts/log-analysis.sh

# 3. Apply security patches
./security/scripts/apply-security-updates.sh

# 4. Restore from clean backup
./disaster-recovery/scripts/restore-from-backup.sh --security-incident
```

## 🔄 Recovery Time Objectives (RTO)

| Incident Type | Target RTO | Target RPO |
|---------------|------------|------------|
| Complete failure | 15 minutes | 5 minutes |
| Database issues | 30 minutes | 15 minutes |
| Partial degradation | 5 minutes | 1 minute |
| Security incident | 2 hours | 1 hour |

## 📊 Post-Incident Procedures

1. **Document the incident** - Create detailed incident report
2. **Conduct post-mortem** - Analyze root cause and timeline
3. **Update procedures** - Improve based on lessons learned
4. **Test recovery** - Validate all recovery procedures
5. **Communicate status** - Update stakeholders on resolution

## 🧪 Testing Schedule

- **Weekly**: Backup restoration tests
- **Monthly**: Failover procedure tests  
- **Quarterly**: Full disaster recovery simulation
- **Annually**: Complete infrastructure rebuild test
EOF

    cat > "${DR_DIR}/runbooks/backup-restore-procedures.md" << 'EOF'
# Backup and Restore Procedures

## 📦 Backup Types and Schedule

### Automated Backups
- **Database**: Every 6 hours, retained for 30 days
- **Application data**: Daily, retained for 7 days
- **Full system**: Weekly, retained for 4 weeks
- **Configuration**: After each deployment

### Backup Locations
- **Primary**: Local storage (`/backups`)
- **Secondary**: Remote storage (S3/GCS)
- **Tertiary**: Offsite tape storage (monthly)

## 🔄 Restore Procedures

### Database Restore
```bash
# List available backups
ls -la /backups/daily/

# Restore specific backup
./disaster-recovery/scripts/database-restore.sh /backups/daily/claude_tui_20250825_120000.sql.gz

# Validate restore
./scripts/validate-database.sh
```

### Full System Restore
```bash
# Prepare clean environment
./disaster-recovery/scripts/prepare-clean-environment.sh

# Restore from backup
./disaster-recovery/scripts/full-system-restore.sh /backups/full-system/20250825_120000

# Validate system
./scripts/production-validation.sh
```

## ✅ Backup Validation

All backups are automatically validated:
- **Integrity check**: SHA256 checksums verified
- **Restoration test**: Monthly test restores performed
- **Data validation**: Schema and data consistency checks
- **Performance test**: Restore time benchmarking
EOF

    log INFO "Disaster recovery runbooks created"
}

# Setup monitoring integration
setup_dr_monitoring() {
    log INFO "Setting up disaster recovery monitoring..."
    
    cat > "${DR_DIR}/configs/dr-prometheus-rules.yml" << 'EOF'
groups:
- name: disaster-recovery
  rules:
  - alert: BackupFailed
    expr: increase(backup_failed_total[1h]) > 0
    for: 0m
    labels:
      severity: critical
      component: backup
    annotations:
      summary: "Backup process failed"
      description: "{{ $labels.backup_type }} backup failed: {{ $labels.error }}"
      runbook_url: "https://docs.claude-tui.dev/runbooks/backup-failure"

  - alert: DisasterRecoveryTestOverdue
    expr: time() - dr_test_last_success_timestamp > 604800  # 7 days
    for: 0m
    labels:
      severity: warning
      component: dr-testing
    annotations:
      summary: "DR test overdue"
      description: "Disaster recovery test hasn't been run in over 7 days"

  - alert: FailoverTriggered
    expr: increase(failover_triggered_total[5m]) > 0
    for: 0m
    labels:
      severity: critical
      component: failover
    annotations:
      summary: "Automatic failover triggered"
      description: "System failed over from {{ $labels.from }} to {{ $labels.to }}"

  - alert: BackupStorageAlmostFull
    expr: (backup_storage_used_bytes / backup_storage_total_bytes) > 0.85
    for: 5m
    labels:
      severity: warning
      component: storage
    annotations:
      summary: "Backup storage almost full"
      description: "Backup storage is {{ $value | humanizePercentage }} full"
EOF

    log INFO "Disaster recovery monitoring setup completed"
}

# Main execution
main() {
    log INFO "Starting Disaster Recovery setup..."
    
    setup_backup_infrastructure
    setup_dr_automation
    create_dr_runbooks
    setup_dr_monitoring
    
    log INFO "Disaster Recovery setup completed!"
    
    echo
    echo "🎉 Disaster Recovery System is ready!"
    echo
    echo "✅ Configured Components:"
    echo "  - Automated database backups"
    echo "  - Full system backup procedures"
    echo "  - Health monitoring and failover"
    echo "  - Comprehensive recovery runbooks"
    echo "  - DR-specific monitoring and alerting"
    echo
    echo "🔧 Next Steps:"
    echo "1. Configure backup retention policies"
    echo "2. Set up remote backup storage"
    echo "3. Enable failover monitoring service"
    echo "4. Schedule monthly DR tests"
    echo "5. Review and customize runbooks"
    echo
    echo "📋 Critical Actions Required:"
    echo "  - Update emergency contact information"
    echo "  - Configure Slack webhook for alerts"
    echo "  - Set up DNS update scripts"
    echo "  - Test backup and restore procedures"
    echo
    echo "🚨 For 100% uptime:"
    echo "  - Deploy to multiple availability zones"
    echo "  - Configure geographic load balancing"  
    echo "  - Enable real-time data replication"
    echo "  - Set up 24/7 monitoring alerts"
    echo
}

# Run main function
main "$@"