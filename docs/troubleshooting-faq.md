# Claude-TIU Troubleshooting Guide & FAQ

## 🔧 Complete Problem-Solving Reference

This comprehensive troubleshooting guide provides solutions for common issues, performance problems, and frequently asked questions about Claude-TIU.

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Authentication Problems](#authentication-problems)
4. [API Connection Issues](#api-connection-issues)
5. [Performance Problems](#performance-problems)
6. [TUI Interface Issues](#tui-interface-issues)
7. [Docker & Deployment Issues](#docker--deployment-issues)
8. [Database Problems](#database-problems)
9. [AI & Validation Issues](#ai--validation-issues)
10. [Frequently Asked Questions](#frequently-asked-questions)
11. [Error Code Reference](#error-code-reference)
12. [Getting Help](#getting-help)

---

## Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system diagnostics
claude-tiu doctor --comprehensive

# Expected output:
✅ System Requirements: All met
✅ Python Environment: 3.11.0 (OK)
✅ Node.js Environment: 18.17.0 (OK)
✅ Network Connectivity: Good (120ms to API)
✅ API Key: Valid and active
✅ Claude Flow: Connected and responsive
✅ Database: PostgreSQL 15.3 (Connected)
✅ Redis Cache: 7.0.11 (Connected)
✅ Disk Space: 45.2GB free (OK)
✅ Memory: 8.1GB available (OK)
⚠️ High CPU usage detected (85%)
💡 Recommendation: Close unnecessary applications
```

### Quick Status Commands
```bash
# Check Claude-TIU status
claude-tiu status

# Test API connectivity
claude-tiu test-connection

# View active processes
claude-tiu ps

# Check logs
claude-tiu logs --tail 50

# Memory usage
claude-tiu memory-usage

# Clear caches
claude-tiu cache clear
```

---

## Installation Issues

### Issue: "Command not found: claude-tiu"

**Symptoms:**
```bash
$ claude-tiu --version
bash: claude-tiu: command not found
```

**Solution:**
```bash
# Option 1: Add to PATH
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Option 2: Create symlink
sudo ln -s ~/.local/bin/claude-tiu /usr/local/bin/claude-tiu

# Option 3: Use python -m
python -m claude_tiu --version

# Option 4: Reinstall with --user flag
pip install --user claude-tiu

# Option 5: Install globally (if you have admin rights)
sudo pip install claude-tiu
```

### Issue: Permission Denied Errors

**Symptoms:**
```bash
$ claude-tiu create my-project
PermissionError: [Errno 13] Permission denied: '/usr/local/bin/claude-tiu'
```

**Solution:**
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/.local/

# Fix permissions
chmod +x ~/.local/bin/claude-tiu

# Alternative: Use virtual environment
python -m venv claude-tiu-env
source claude-tiu-env/bin/activate
pip install claude-tiu
```

### Issue: Python Version Conflicts

**Symptoms:**
```bash
$ claude-tiu --version
ModuleNotFoundError: No module named 'claude_tiu'
```

**Solution:**
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.11 -m pip install claude-tiu
python3.11 -m claude_tiu --version

# Use pyenv for version management
pyenv install 3.11.0
pyenv global 3.11.0
pip install claude-tiu
```

### Issue: SSL Certificate Errors

**Symptoms:**
```bash
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
```bash
# Update certificates
pip install --upgrade certifi

# For macOS
/Applications/Python\ 3.11/Install\ Certificates.command

# For corporate networks
pip install --trusted-host pypi.org --trusted-host pypi.python.org claude-tiu

# Set certificate bundle
export SSL_CERT_FILE=$(python -m certifi)
```

### Issue: Dependency Conflicts

**Symptoms:**
```bash
ERROR: pip's dependency resolver does not currently support
```

**Solution:**
```bash
# Create clean virtual environment
python -m venv clean-env
source clean-env/bin/activate
pip install --upgrade pip setuptools wheel

# Install Claude-TIU
pip install claude-tiu

# If conflicts persist, use conda
conda create -n claude-tiu python=3.11
conda activate claude-tiu
pip install claude-tiu
```

---

## Authentication Problems

### Issue: Invalid API Key

**Symptoms:**
```bash
$ claude-tiu test-connection
❌ Authentication failed: Invalid API key
```

**Solution:**
```bash
# Check current API key
echo $CLAUDE_API_KEY

# Verify key format (should start with 'sk-')
# Get new key from https://claude-tiu.dev/dashboard

# Set API key
export CLAUDE_API_KEY="sk-your-actual-key-here"

# Make permanent
echo 'export CLAUDE_API_KEY="sk-your-key"' >> ~/.bashrc
source ~/.bashrc

# Or use config command
claude-tiu config set api-key sk-your-key

# Test connection
claude-tiu test-connection
```

### Issue: Token Expired

**Symptoms:**
```json
{
  "error": "Token expired",
  "code": "AUTH_TOKEN_EXPIRED"
}
```

**Solution:**
```bash
# Refresh authentication
claude-tiu auth refresh

# Or re-login
claude-tiu auth login

# Clear cached tokens
claude-tiu cache clear --auth-only

# Check token expiry
claude-tiu auth status
```

### Issue: MFA Problems

**Symptoms:**
```bash
❌ MFA verification failed
```

**Solution:**
```bash
# Check time sync (important for TOTP)
sudo ntpdate -s time.nist.gov

# Generate backup codes
claude-tiu auth backup-codes

# Disable MFA temporarily (if you have backup codes)
claude-tiu auth disable-mfa --backup-code YOUR_CODE

# Re-enable MFA
claude-tiu auth enable-mfa

# Use different MFA method
claude-tiu auth mfa --method sms
```

---

## API Connection Issues

### Issue: Connection Timeouts

**Symptoms:**
```bash
❌ Request timeout after 30 seconds
```

**Solution:**
```bash
# Check network connectivity
ping api.claude-tiu.dev
curl -I https://api.claude-tiu.dev/health

# Increase timeout
claude-tiu config set request-timeout 60

# Check for proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Test with curl
curl -H "Authorization: Bearer $CLAUDE_API_KEY" \
     https://api.claude-tiu.dev/v1/health
```

### Issue: Rate Limit Exceeded

**Symptoms:**
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

**Solution:**
```bash
# Check rate limit status
claude-tiu rate-limits

# Wait for reset
sleep 60

# Enable automatic retry
claude-tiu config set auto-retry true
claude-tiu config set retry-delay 5

# Upgrade plan for higher limits
claude-tiu billing upgrade
```

### Issue: SSL/TLS Problems

**Symptoms:**
```bash
SSL handshake failed
```

**Solution:**
```bash
# Check SSL certificates
openssl s_client -connect api.claude-tiu.dev:443 -servername api.claude-tiu.dev

# Update CA certificates
sudo apt-get update && sudo apt-get install ca-certificates

# For corporate firewalls
export REQUESTS_CA_BUNDLE=/path/to/your/certificate.pem

# Disable SSL verification (not recommended)
claude-tiu config set ssl-verify false
```

---

## Performance Problems

### Issue: Slow AI Response Times

**Symptoms:**
- Code generation takes >60 seconds
- Validation processes hang
- TUI becomes unresponsive

**Diagnosis:**
```bash
# Check AI performance metrics
claude-tiu metrics ai-performance

# Monitor resource usage
claude-tiu monitor --real-time

# Check AI service health
claude-tiu ai-status
```

**Solution:**
```bash
# Optimize AI settings
claude-tiu config set ai-creativity 0.6  # Lower creativity = faster responses
claude-tiu config set ai-timeout 45      # Reasonable timeout
claude-tiu config set parallel-requests 3 # Limit concurrent requests

# Clear AI cache
claude-tiu cache clear --ai-only

# Use local processing where possible
claude-tiu config set prefer-local true

# Enable performance mode
claude-tiu config set performance-mode high
```

### Issue: High Memory Usage

**Symptoms:**
```bash
$ claude-tiu memory-usage
Memory usage: 4.2GB (85% of available)
⚠️ High memory usage detected
```

**Solution:**
```bash
# Enable memory optimization
claude-tiu config set memory-optimization true

# Reduce cache size
claude-tiu config set cache-size 256MB

# Enable garbage collection tuning
claude-tiu config set gc-optimization true

# Restart with memory limits
claude-tiu restart --memory-limit 2GB

# Monitor memory usage
claude-tiu monitor memory --duration 300
```

### Issue: Database Performance Problems

**Symptoms:**
- Slow project loading
- Timeout errors on large operations
- Database connection pool exhausted

**Solution:**
```bash
# Check database performance
claude-tiu db performance

# Optimize database
claude-tiu db optimize

# Rebuild indexes
claude-tiu db reindex

# Check connection pool
claude-tiu config set db-pool-size 20

# Enable connection pooling
claude-tiu config set db-pool-enabled true
```

---

## TUI Interface Issues

### Issue: Interface Not Rendering Correctly

**Symptoms:**
- Garbled text
- Missing borders
- Overlapping elements

**Solution:**
```bash
# Check terminal compatibility
echo $TERM
tput colors

# Set compatible terminal type
export TERM=xterm-256color

# Force UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Try different rendering mode
claude-tiu --render-mode basic

# Reset terminal
reset

# Use safe mode
claude-tiu --safe-mode
```

### Issue: Keyboard Shortcuts Not Working

**Symptoms:**
- Function keys not responding
- Alt combinations not working
- Special characters not typing

**Solution:**
```bash
# Check terminal key mapping
claude-tiu test-keys

# Reset keyboard shortcuts
claude-tiu config reset-shortcuts

# Use alternative shortcuts
claude-tiu --use-alt-shortcuts

# Check terminal settings
stty -a

# Test in different terminal
# Try: gnome-terminal, xterm, iTerm2, Windows Terminal
```

### Issue: Display Scaling Problems

**Symptoms:**
- Text too small or too large
- Interface elements cut off
- Scrolling issues

**Solution:**
```bash
# Adjust interface scale
claude-tiu config set ui-scale 1.2

# Set font size
claude-tiu config set font-size 14

# Force terminal size
stty rows 40 cols 120

# Use responsive mode
claude-tiu --responsive-ui

# Check display DPI
xrandr | grep -E " connected (primary )?[0-9]+"
```

---

## Docker & Deployment Issues

### Issue: Container Won't Start

**Symptoms:**
```bash
$ docker-compose up
ERROR: Container failed to start
```

**Solution:**
```bash
# Check container logs
docker-compose logs claude-tiu

# Check resource limits
docker stats

# Increase memory limits
# In docker-compose.yml:
services:
  claude-tiu:
    mem_limit: 2g
    mem_reservation: 1g

# Check port conflicts
netstat -tulpn | grep :8000

# Clean Docker state
docker system prune -a
docker volume prune
```

### Issue: Database Connection in Docker

**Symptoms:**
```bash
FATAL: could not connect to server: No such file or directory
```

**Solution:**
```bash
# Wait for database to be ready
# In docker-compose.yml:
services:
  claude-tiu:
    depends_on:
      db:
        condition: service_healthy
    
  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

# Check network connectivity
docker exec claude-tiu ping db

# Verify environment variables
docker exec claude-tiu env | grep DATABASE_URL
```

### Issue: Kubernetes Deployment Problems

**Symptoms:**
```bash
$ kubectl get pods
claude-tiu-xxx   0/1   CrashLoopBackOff
```

**Solution:**
```bash
# Check pod logs
kubectl logs claude-tiu-xxx -f

# Describe pod for events
kubectl describe pod claude-tiu-xxx

# Check resource limits
kubectl top pod claude-tiu-xxx

# Verify config maps and secrets
kubectl get configmap claude-tiu-config -o yaml
kubectl get secret claude-tiu-secrets -o yaml

# Check service account permissions
kubectl auth can-i --list --as=system:serviceaccount:default:claude-tiu

# Scale down and up
kubectl scale deployment claude-tiu --replicas=0
kubectl scale deployment claude-tiu --replicas=3
```

---

## Database Problems

### Issue: Migration Failures

**Symptoms:**
```bash
$ claude-tiu db migrate
alembic.util.exc.CommandError: Can't locate revision identified by 'xyz123'
```

**Solution:**
```bash
# Check migration history
claude-tiu db history

# Reset to specific revision
claude-tiu db downgrade base
claude-tiu db upgrade head

# Force migration
claude-tiu db stamp head
claude-tiu db upgrade

# Backup before fixing
claude-tiu db backup --file backup_before_fix.sql

# Manual migration repair
psql $DATABASE_URL -c "SELECT version_num FROM alembic_version;"
```

### Issue: Database Connection Pool Exhausted

**Symptoms:**
```bash
sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 0 reached
```

**Solution:**
```bash
# Increase pool size
claude-tiu config set db-pool-size 50
claude-tiu config set db-max-overflow 20

# Check for connection leaks
claude-tiu db connections

# Enable connection recycling
claude-tiu config set db-pool-recycle 3600

# Restart application
claude-tiu restart
```

### Issue: Slow Queries

**Symptoms:**
- Long response times
- Database CPU high
- Query timeouts

**Solution:**
```bash
# Enable query logging
claude-tiu config set db-log-queries true

# Analyze slow queries
claude-tiu db slow-queries

# Update statistics
claude-tiu db analyze

# Rebuild indexes
claude-tiu db reindex

# Check for missing indexes
claude-tiu db index-suggestions
```

---

## AI & Validation Issues

### Issue: AI Generation Produces Invalid Code

**Symptoms:**
- Syntax errors in generated code
- Missing imports
- Incomplete functions

**Solution:**
```bash
# Increase validation strictness
claude-tiu config set validation-level strict

# Enable auto-fix
claude-tiu config set auto-fix true

# Use better prompting
claude-tiu config set ai-creativity 0.7
claude-tiu config set include-context true

# Enable post-generation validation
claude-tiu config set post-validation true

# Use specific language models
claude-tiu ai-config set model claude-3-sonnet
```

### Issue: Validation Engine False Positives

**Symptoms:**
- Valid code marked as placeholder
- Working code flagged as incomplete
- High-quality code gets low scores

**Solution:**
```bash
# Tune validation sensitivity
claude-tiu config set validation-sensitivity 0.8

# Update validation models
claude-tiu update --validation-models

# Train custom patterns
claude-tiu validation train-custom --data your_training_data.json

# Check validation accuracy
claude-tiu validation benchmark

# Use manual validation mode
claude-tiu validate --manual-review
```

### Issue: Anti-Hallucination Detection Too Aggressive

**Symptoms:**
- Valid implementations flagged as placeholders
- Legitimate TODO comments cause failures
- Generated code rejected unnecessarily

**Solution:**
```bash
# Adjust detection threshold
claude-tiu config set anti-hallucination-threshold 0.9

# Whitelist patterns
claude-tiu validation whitelist-pattern "# TODO: implement feature X"

# Disable aggressive detection
claude-tiu config set validation-aggressive false

# Custom validation rules
claude-tiu validation add-rule --file custom_rules.yaml

# Review detection accuracy
claude-tiu validation accuracy-report
```

---

## Frequently Asked Questions

### General Usage

**Q: How do I upgrade Claude-TIU to the latest version?**
```bash
# Upgrade via pip
pip install --upgrade claude-tiu

# Or via package manager
apt update && apt upgrade claude-tiu

# Check version
claude-tiu --version

# Update configuration if needed
claude-tiu config migrate
```

**Q: Can I use Claude-TIU offline?**
```bash
# Limited offline functionality available
claude-tiu --offline-mode

# Pre-download models for offline use
claude-tiu download-models --all

# Enable local processing
claude-tiu config set prefer-local true
```

**Q: How do I backup my projects and settings?**
```bash
# Backup projects
claude-tiu backup projects --output projects_backup.tar.gz

# Backup settings
claude-tiu config export --file config_backup.yaml

# Full system backup
claude-tiu backup --full --output full_backup_$(date +%Y%m%d).tar.gz

# Restore from backup
claude-tiu restore --file full_backup_20240115.tar.gz
```

### Performance & Scaling

**Q: How can I improve AI response times?**
```bash
# Optimize AI settings
claude-tiu config set ai-creativity 0.6
claude-tiu config set ai-timeout 30
claude-tiu config set cache-ai-responses true

# Use performance mode
claude-tiu --performance-mode

# Enable parallel processing
claude-tiu config set parallel-ai-requests 3
```

**Q: What are the system requirements for enterprise use?**
```yaml
enterprise_requirements:
  minimum:
    cpu: "8 cores, 3.0GHz+"
    memory: "32GB"
    storage: "500GB SSD"
    network: "1Gbps"
  
  recommended:
    cpu: "16 cores, 3.5GHz+"
    memory: "64GB"
    storage: "1TB NVMe"
    network: "10Gbps"
    
  clustering:
    nodes: "3+ nodes"
    load_balancer: "Required"
    database: "PostgreSQL cluster"
    cache: "Redis cluster"
```

### Security & Compliance

**Q: Is my code and data secure?**
```yaml
security_measures:
  encryption:
    at_rest: "AES-256"
    in_transit: "TLS 1.3"
    key_management: "Enterprise KMS"
    
  compliance:
    - "SOC 2 Type II"
    - "ISO 27001" 
    - "GDPR compliant"
    - "CCPA compliant"
    
  data_handling:
    retention: "User controlled"
    deletion: "Secure deletion available"
    portability: "Data export supported"
```

**Q: Can I run Claude-TIU in my corporate network?**
```bash
# Configure for corporate proxy
claude-tiu config set proxy http://proxy.company.com:8080

# Use corporate certificates
claude-tiu config set ca-bundle /path/to/corporate/ca.pem

# Enable LDAP/AD integration
claude-tiu auth configure --type ldap --server ldap.company.com

# Air-gapped deployment
claude-tiu deploy --air-gapped --models-bundle local_models.tar.gz
```

### Integration & API

**Q: How do I integrate Claude-TIU with my existing tools?**
```python
# Python integration
from claude_tiu import ClaudeTIU, ProjectConfig

client = ClaudeTIU(api_key="your-key")
project = await client.create_project(
    ProjectConfig(name="integration-test", type="fastapi")
)

# REST API integration
import requests

response = requests.post(
    "https://api.claude-tiu.dev/v1/projects",
    headers={"Authorization": "Bearer your-token"},
    json={"name": "api-project", "type": "react"}
)

# CLI integration
claude-tiu create my-project --type django --output-json | jq .project_id
```

**Q: Can I extend Claude-TIU with custom plugins?**
```python
# Create custom plugin
from claude_tiu.plugins import Plugin, PluginManager

class MyCustomPlugin(Plugin):
    name = "my-custom-plugin"
    version = "1.0.0"
    
    async def on_project_created(self, project):
        # Custom logic when project is created
        await self.send_notification(f"Project {project.name} created")
    
    async def custom_command(self, args):
        # Custom CLI command
        return {"status": "success", "message": "Custom command executed"}

# Register plugin
manager = PluginManager()
manager.register(MyCustomPlugin())
```

### Troubleshooting Workflows

**Q: My validation is failing on valid code. What should I do?**
```bash
# Step 1: Check validation settings
claude-tiu config get validation-level

# Step 2: Run with detailed output
claude-tiu validate --verbose --debug my_file.py

# Step 3: Check for false positives
claude-tiu validation analyze-false-positives

# Step 4: Adjust thresholds if needed
claude-tiu config set validation-sensitivity 0.7

# Step 5: Report issues to improve the system
claude-tiu report-issue --type validation --file my_file.py
```

---

## Error Code Reference

### Authentication Errors (AUTH_*)
| Code | Description | Solution |
|------|-------------|----------|
| `AUTH_INVALID_KEY` | API key is invalid or malformed | Check API key format and regenerate if needed |
| `AUTH_TOKEN_EXPIRED` | JWT token has expired | Run `claude-tiu auth refresh` |
| `AUTH_INSUFFICIENT_PERMISSIONS` | User lacks required permissions | Contact admin to update user role |
| `AUTH_MFA_REQUIRED` | Multi-factor authentication required | Complete MFA setup with `claude-tiu auth mfa` |
| `AUTH_RATE_LIMITED` | Too many authentication attempts | Wait before retrying |

### Network Errors (NET_*)
| Code | Description | Solution |
|------|-------------|----------|
| `NET_CONNECTION_TIMEOUT` | Request timed out | Check network and increase timeout |
| `NET_DNS_RESOLUTION_FAILED` | Cannot resolve API hostname | Check DNS settings |
| `NET_SSL_ERROR` | SSL/TLS connection failed | Update certificates or check proxy |
| `NET_PROXY_ERROR` | Proxy server error | Verify proxy configuration |
| `NET_RATE_LIMITED` | API rate limit exceeded | Wait or upgrade plan |

### Validation Errors (VAL_*)
| Code | Description | Solution |
|------|-------------|----------|
| `VAL_PLACEHOLDER_DETECTED` | Code contains placeholders | Complete implementation |
| `VAL_SYNTAX_ERROR` | Syntax errors in code | Fix syntax issues |
| `VAL_SECURITY_ISSUE` | Security vulnerability found | Address security concerns |
| `VAL_PERFORMANCE_ISSUE` | Performance problems detected | Optimize code performance |
| `VAL_ENGINE_ERROR` | Validation engine failure | Report issue to support |

### System Errors (SYS_*)
| Code | Description | Solution |
|------|-------------|----------|
| `SYS_INSUFFICIENT_MEMORY` | Not enough memory available | Free memory or increase limits |
| `SYS_DISK_FULL` | Disk space exhausted | Free disk space |
| `SYS_PERMISSION_DENIED` | Permission denied for operation | Check file/directory permissions |
| `SYS_PROCESS_KILLED` | Process was terminated | Check system resources |
| `SYS_CONFIG_ERROR` | Configuration file error | Validate and fix configuration |

---

## Getting Help

### Self-Service Resources
1. **Documentation**: https://docs.claude-tiu.dev
2. **API Reference**: https://api.claude-tiu.dev/docs
3. **Community Forum**: https://community.claude-tiu.dev
4. **Knowledge Base**: https://help.claude-tiu.dev
5. **Status Page**: https://status.claude-tiu.dev

### Debug Information Collection
```bash
# Generate comprehensive debug report
claude-tiu debug-report --include-logs --include-config

# This creates a file: claude-tiu-debug-YYYYMMDD-HHMMSS.tar.gz
# Upload this file when requesting support
```

### Contact Support
- **Community Support**: Discord, Forums, GitHub Issues
- **Email Support**: support@claude-tiu.dev
- **Enterprise Support**: enterprise@claude-tiu.dev
- **Security Issues**: security@claude-tiu.dev

### Support Ticket Template
```
**Problem Description:**
Brief description of the issue

**Environment:**
- Claude-TIU Version: X.X.X
- Operating System: 
- Python Version:
- Database: 

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Error Messages:**
Complete error output

**Debug Report:**
Attach claude-tiu-debug-*.tar.gz file
```

### Emergency Support
For production-critical issues:
- **Phone**: +1-800-CLAUDE-TIU
- **Slack**: #claude-tiu-emergency (Enterprise customers)
- **Priority**: Include "URGENT" in email subject

---

## Best Practices for Troubleshooting

### 1. Start with Basics
- Check system requirements
- Verify network connectivity  
- Update to latest version
- Clear caches

### 2. Gather Information
- Enable debug logging
- Collect error messages
- Note environment details
- Document reproduction steps

### 3. Systematic Approach
- Test in isolation
- Change one variable at a time
- Use process of elimination
- Document what works/doesn't work

### 4. Use Available Tools
```bash
# Monitoring
claude-tiu monitor --real-time

# Debugging
claude-tiu --debug --verbose

# Testing
claude-tiu test-all

# Profiling
claude-tiu profile --duration 60
```

### 5. Know When to Escalate
- Security-related issues
- Data corruption concerns
- Production downtime
- Compliance violations

---

**Need more help? We're here for you! 🚀**

*This troubleshooting guide is regularly updated based on user feedback and common issues. For the latest version, visit: https://docs.claude-tiu.dev/troubleshooting*