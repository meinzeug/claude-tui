#!/bin/bash

# Database Migration Script for Claude-TIU Production Deployment
# Handles schema migrations, data migrations, and rollbacks

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-production}"
APP_NAME="${APP_NAME:-claude-tiu}"
MIGRATION_IMAGE="${MIGRATION_IMAGE:-ghcr.io/claude-tiu/claude-tiu:latest}"
MIGRATION_TIMEOUT="${MIGRATION_TIMEOUT:-300}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check if database connection secret exists
    if ! kubectl get secret claude-tiu-secrets -n "$NAMESPACE" &> /dev/null; then
        error "Database secrets not found in namespace $NAMESPACE"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Function to create database backup
create_backup() {
    local backup_name="pre-migration-backup-$(date +%Y%m%d-%H%M%S)"
    
    log "Creating database backup: $backup_name"
    
    # Create backup job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-backup-$backup_name
  namespace: $NAMESPACE
  labels:
    app: claude-tiu-migration
    type: backup
spec:
  ttlSecondsAfterFinished: 86400  # 24 hours
  template:
    metadata:
      labels:
        app: claude-tiu-migration
        type: backup
    spec:
      restartPolicy: Never
      containers:
      - name: db-backup
        image: postgres:15-alpine
        command:
        - /bin/sh
        - -c
        - |
          set -e
          echo "Starting database backup..."
          pg_dump "\$DATABASE_URL" > /backup/\$BACKUP_NAME.sql
          echo "Backup completed successfully"
          ls -la /backup/
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: database-url
        - name: BACKUP_NAME
          value: "$backup_name"
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: database-password
              optional: true
        volumeMounts:
        - name: backup-storage
          mountPath: /backup
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: backup-storage
        persistentVolumeClaim:
          claimName: claude-tiu-backup-pvc
EOF

    # Wait for backup job to complete
    log "Waiting for backup to complete..."
    if ! kubectl wait --for=condition=complete job/db-backup-$backup_name -n "$NAMESPACE" --timeout="${MIGRATION_TIMEOUT}s"; then
        error "Database backup failed"
        kubectl logs -l job-name=db-backup-$backup_name -n "$NAMESPACE" --tail=50
        return 1
    fi
    
    log "Database backup completed successfully: $backup_name"
    echo "$backup_name"
}

# Function to run database migration
run_migration() {
    local migration_type="${1:-up}"
    local target_version="${2:-latest}"
    
    log "Running database migration: $migration_type to $target_version"
    
    # Create migration job
    local job_name="db-migration-$(date +%Y%m%d-%H%M%S)"
    
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $job_name
  namespace: $NAMESPACE
  labels:
    app: claude-tiu-migration
    type: migration
    migration-type: $migration_type
spec:
  ttlSecondsAfterFinished: 86400  # 24 hours
  template:
    metadata:
      labels:
        app: claude-tiu-migration
        type: migration
    spec:
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: migration
        image: $MIGRATION_IMAGE
        command:
        - /bin/sh
        - -c
        - |
          set -e
          echo "Starting database migration..."
          echo "Migration type: $migration_type"
          echo "Target version: $target_version"
          
          # Run database migrations using your migration tool
          # This is a placeholder - replace with your actual migration command
          case "$migration_type" in
            "up")
              echo "Running schema migrations..."
              python -m alembic upgrade $target_version
              ;;
            "down")
              echo "Rolling back migrations..."
              python -m alembic downgrade $target_version
              ;;
            "status")
              echo "Checking migration status..."
              python -m alembic current
              python -m alembic history
              ;;
            "validate")
              echo "Validating database schema..."
              python -m alembic check
              ;;
            *)
              echo "Unknown migration type: $migration_type"
              exit 1
              ;;
          esac
          
          echo "Migration completed successfully"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: database-url
        - name: CLAUDE_TIU_ENV
          value: "production"
        - name: PYTHONPATH
          value: "/app/src"
        - name: MIGRATION_TYPE
          value: "$migration_type"
        - name: TARGET_VERSION
          value: "$target_version"
        volumeMounts:
        - name: migration-logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: migration-logs
        emptyDir: {}
EOF

    # Wait for migration job to complete
    log "Waiting for migration to complete..."
    if ! kubectl wait --for=condition=complete job/$job_name -n "$NAMESPACE" --timeout="${MIGRATION_TIMEOUT}s"; then
        error "Database migration failed"
        echo "Migration job logs:"
        kubectl logs -l job-name=$job_name -n "$NAMESPACE" --tail=100
        
        # Check if job failed
        if kubectl get job $job_name -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep -q "True"; then
            error "Migration job failed permanently"
            return 1
        fi
        
        return 1
    fi
    
    log "Database migration completed successfully"
    
    # Show migration logs
    info "Migration job logs:"
    kubectl logs -l job-name=$job_name -n "$NAMESPACE" --tail=20
    
    return 0
}

# Function to verify migration
verify_migration() {
    log "Verifying database migration..."
    
    # Create verification job
    local job_name="db-verify-$(date +%Y%m%d-%H%M%S)"
    
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $job_name
  namespace: $NAMESPACE
  labels:
    app: claude-tiu-migration
    type: verification
spec:
  ttlSecondsAfterFinished: 3600  # 1 hour
  template:
    metadata:
      labels:
        app: claude-tiu-migration
        type: verification
    spec:
      restartPolicy: Never
      containers:
      - name: verify
        image: $MIGRATION_IMAGE
        command:
        - /bin/sh
        - -c
        - |
          set -e
          echo "Verifying database state..."
          
          # Run verification checks
          python -c "
          import sys
          import asyncpg
          import asyncio
          import os
          
          async def verify_database():
              try:
                  conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
                  
                  # Check if tables exist
                  tables = await conn.fetch('''
                      SELECT table_name 
                      FROM information_schema.tables 
                      WHERE table_schema = 'public'
                  ''')
                  
                  print(f'Found {len(tables)} tables in database')
                  for table in tables:
                      print(f'  - {table[\"table_name\"]}')
                  
                  # Check migration status
                  try:
                      version_info = await conn.fetchrow('SELECT * FROM alembic_version')
                      print(f'Current migration version: {version_info[\"version_num\"]}')
                  except:
                      print('No migration version table found')
                  
                  # Basic connectivity test
                  result = await conn.fetchval('SELECT 1')
                  if result == 1:
                      print('Database connectivity: OK')
                  else:
                      print('Database connectivity: FAILED')
                      sys.exit(1)
                  
                  await conn.close()
                  print('Database verification completed successfully')
                  
              except Exception as e:
                  print(f'Database verification failed: {str(e)}')
                  sys.exit(1)
          
          asyncio.run(verify_database())
          "
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: claude-tiu-secrets
              key: database-url
        - name: PYTHONPATH
          value: "/app/src"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
EOF

    # Wait for verification to complete
    if ! kubectl wait --for=condition=complete job/$job_name -n "$NAMESPACE" --timeout="60s"; then
        error "Database verification failed"
        kubectl logs -l job-name=$job_name -n "$NAMESPACE" --tail=50
        return 1
    fi
    
    log "Database verification completed successfully"
    kubectl logs -l job-name=$job_name -n "$NAMESPACE" --tail=10
    
    return 0
}

# Function to rollback migration
rollback_migration() {
    local target_version="${1:-HEAD-1}"
    
    warn "Rolling back database migration to version: $target_version"
    
    # Confirm rollback
    read -p "Are you sure you want to rollback the database? This action may cause data loss. (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        info "Rollback cancelled"
        return 0
    fi
    
    # Create backup before rollback
    local backup_name
    backup_name=$(create_backup)
    if [[ $? -ne 0 ]]; then
        error "Failed to create backup before rollback"
        return 1
    fi
    
    # Run rollback migration
    if run_migration "down" "$target_version"; then
        log "Database rollback completed successfully"
        verify_migration
    else
        error "Database rollback failed"
        return 1
    fi
}

# Function to show migration status
show_status() {
    log "Checking database migration status..."
    
    # Run status check
    if run_migration "status"; then
        log "Migration status check completed"
    else
        error "Failed to check migration status"
        return 1
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old database backups (older than $BACKUP_RETENTION_DAYS days)..."
    
    # Get backup PVC and list old backups
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: cleanup-backups-$(date +%Y%m%d-%H%M%S)
  namespace: $NAMESPACE
  labels:
    app: claude-tiu-migration
    type: cleanup
spec:
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: claude-tiu-migration
        type: cleanup
    spec:
      restartPolicy: Never
      containers:
      - name: cleanup
        image: alpine:latest
        command:
        - /bin/sh
        - -c
        - |
          set -e
          echo "Cleaning up old backups..."
          
          # Find and remove backups older than retention period
          find /backup -name "*.sql" -mtime +$BACKUP_RETENTION_DAYS -type f -print
          find /backup -name "*.sql" -mtime +$BACKUP_RETENTION_DAYS -type f -delete
          
          echo "Backup cleanup completed"
          df -h /backup
        env:
        - name: BACKUP_RETENTION_DAYS
          value: "$BACKUP_RETENTION_DAYS"
        volumeMounts:
        - name: backup-storage
          mountPath: /backup
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "100m"
      volumes:
      - name: backup-storage
        persistentVolumeClaim:
          claimName: claude-tiu-backup-pvc
EOF

    log "Backup cleanup job started"
}

# Function to create required PVC for backups
ensure_backup_pvc() {
    if ! kubectl get pvc claude-tiu-backup-pvc -n "$NAMESPACE" &> /dev/null; then
        log "Creating backup PVC..."
        
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: claude-tiu-backup-pvc
  namespace: $NAMESPACE
  labels:
    app: claude-tiu
    component: backup
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
EOF
        
        log "Backup PVC created"
    fi
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    migrate [version]         Run database migrations (default: latest)
    rollback [version]        Rollback database to version (default: HEAD-1)
    status                    Show current migration status
    backup                    Create database backup
    verify                    Verify database state
    cleanup                   Cleanup old backups
    help                      Show this help message

Examples:
    $0 migrate                # Migrate to latest version
    $0 migrate 001_initial    # Migrate to specific version
    $0 rollback               # Rollback one version
    $0 rollback 001_initial   # Rollback to specific version
    $0 status                 # Show migration status
    $0 backup                 # Create backup
    $0 verify                 # Verify database
    $0 cleanup                # Cleanup old backups

Environment Variables:
    NAMESPACE                 Kubernetes namespace (default: production)
    MIGRATION_IMAGE          Docker image for migrations (default: latest)
    MIGRATION_TIMEOUT        Timeout in seconds (default: 300)
    BACKUP_RETENTION_DAYS    Backup retention period (default: 7)

EOF
}

# Main script logic
case "${1:-}" in
    migrate)
        check_prerequisites
        ensure_backup_pvc
        
        # Create backup before migration
        backup_name=$(create_backup)
        if [[ $? -ne 0 ]]; then
            error "Failed to create pre-migration backup"
            exit 1
        fi
        
        # Run migration
        if run_migration "up" "${2:-latest}"; then
            verify_migration
            log "Migration completed successfully"
        else
            error "Migration failed"
            warn "Pre-migration backup available: $backup_name"
            exit 1
        fi
        ;;
        
    rollback)
        check_prerequisites
        ensure_backup_pvc
        rollback_migration "${2:-HEAD-1}"
        ;;
        
    status)
        check_prerequisites
        show_status
        ;;
        
    backup)
        check_prerequisites
        ensure_backup_pvc
        create_backup
        ;;
        
    verify)
        check_prerequisites
        verify_migration
        ;;
        
    cleanup)
        check_prerequisites
        ensure_backup_pvc
        cleanup_old_backups
        ;;
        
    help|--help|-h)
        usage
        ;;
        
    *)
        error "Unknown command: ${1:-}"
        usage
        exit 1
        ;;
esac