# Security Implementation Guide

## Quick Security Implementation Checklist

### Immediate Actions (Priority 1) ⚠️

#### 1. Fix Hardcoded Credentials
```bash
# File: scripts/init_database.py
# BEFORE (Line 201):
password="DevAdmin123!",

# AFTER:
password=os.getenv("DEV_ADMIN_PASSWORD", secrets.token_urlsafe(16)),
```

#### 2. Install Security Tools
```bash
# Install dependency scanning
pip install safety bandit semgrep

# Run security scans
safety check --file requirements.txt
bandit -r src/ --format json -o security_report.json
semgrep --config=auto src/
```

#### 3. Enable Security Monitoring
```python
# Add to main application
from src.security.security_monitor import SecurityMonitor

app.add_middleware(SecurityMonitoringMiddleware)
```

### Security Best Practices Implementation

#### 1. Input Validation Enhanced
```python
# Use the comprehensive SecurityInputValidator
from src.security.input_validator import SecurityInputValidator, ThreatLevel

validator = SecurityInputValidator()

@app.middleware("http")
async def validate_input(request: Request, call_next):
    # Validate all incoming data
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            result = validator.validate_user_prompt(body.decode())
            if not result.is_valid or result.threat_level == ThreatLevel.CRITICAL:
                return JSONResponse({"error": "Invalid input detected"}, status_code=400)
    
    response = await call_next(request)
    return response
```

#### 2. Enhanced Authentication
```python
# JWT Token Security
from src.auth.jwt_auth import JWTAuthenticator
from src.auth.security_config import get_security_config

config = get_security_config()
jwt_auth = JWTAuthenticator(
    secret_key=config.jwt.secret_key,
    access_token_expire_minutes=config.jwt.access_token_expire_minutes
)

# Enable session validation
@app.dependency
async def verify_token(token: str = Depends(oauth2_scheme)):
    token_data = await jwt_auth.validate_token(token, verify_session=True)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token_data
```

#### 3. API Key Management
```python
# Secure API key storage
from src.security.api_key_manager import APIKeyManager, KeyType, EncryptionLevel

key_manager = APIKeyManager(
    encryption_level=EncryptionLevel.ENHANCED,
    master_password=os.getenv("MASTER_KEY_PASSWORD")
)

# Store API keys securely
claude_key_id = key_manager.store_api_key(
    service="claude",
    api_key=os.getenv("CLAUDE_API_KEY"),
    key_type=KeyType.CLAUDE,
    description="Claude API integration key"
)

# Retrieve keys securely
claude_key = key_manager.retrieve_api_key(claude_key_id)
```

### Security Middleware Stack

```python
# Complete security middleware setup
from fastapi import FastAPI
from src.api.middleware.security import SecurityMiddleware
from src.api.middleware.rate_limiting import RateLimitingMiddleware
from src.security.input_validator import InputValidationMiddleware

app = FastAPI()

# Security middleware stack (order matters)
app.add_middleware(SecurityMiddleware, max_request_size=10*1024*1024)
app.add_middleware(InputValidationMiddleware)
app.add_middleware(RateLimitingMiddleware)
```

### Environment Configuration

```bash
# .env security configuration
# JWT Security
JWT_SECRET_KEY=your-super-secure-jwt-secret-key-at-least-32-chars
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Database Security
DATABASE_URL=postgresql://user:password@localhost:5432/db?sslmode=require
REDIS_URL=redis://:password@localhost:6379/0?ssl=true

# API Key Security
MASTER_KEY_PASSWORD=your-master-encryption-password
CLAUDE_API_KEY=your-claude-api-key
OPENAI_API_KEY=your-openai-api-key

# Security Settings
ENVIRONMENT=production
DEBUG=false
SECURITY_LEVEL=enhanced
```

### Docker Security Hardening

```dockerfile
# Secure Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install security updates
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Security headers
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Security Configuration

```yaml
# k8s/security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: claude-tiu-security-policy
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### Security Monitoring Setup

```python
# Security monitoring implementation
from src.security.security_monitor import SecurityMonitor
from src.security.anomaly_detector import AnomalyDetector

class SecurityManager:
    def __init__(self):
        self.monitor = SecurityMonitor()
        self.anomaly_detector = AnomalyDetector()
        
    async def process_security_event(self, event):
        # Analyze security event
        risk_level = await self.monitor.analyze_event(event)
        
        if risk_level >= ThreatLevel.HIGH:
            await self.handle_security_incident(event, risk_level)
            
        # Store for analysis
        await self.monitor.log_security_event(event, risk_level)
    
    async def handle_security_incident(self, event, risk_level):
        if risk_level == ThreatLevel.CRITICAL:
            # Block IP, invalidate sessions
            await self.emergency_lockdown(event)
        
        # Alert security team
        await self.send_security_alert(event, risk_level)
```

### Testing Security Implementation

```python
# Security test examples
import pytest
from src.security.input_validator import SecurityInputValidator

class TestSecurityImplementation:
    
    @pytest.fixture
    def validator(self):
        return SecurityInputValidator()
    
    def test_sql_injection_blocked(self, validator):
        malicious_input = "'; DROP TABLE users; --"
        result = validator.validate_user_prompt(malicious_input)
        assert not result.is_valid
        assert "sql_injection" in [t.threat_type for t in result.threats_detected]
    
    def test_xss_blocked(self, validator):
        malicious_input = "<script>alert('XSS')</script>"
        result = validator.validate_user_prompt(malicious_input)
        assert not result.is_valid
        assert "xss" in [t.threat_type for t in result.threats_detected]
        
    def test_command_injection_blocked(self, validator):
        malicious_input = "test; rm -rf /"
        result = validator.validate_command(malicious_input)
        assert not result.is_valid
```

### CI/CD Security Integration

```yaml
# .github/workflows/security.yml
name: Security Checks
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
          
      - name: Install dependencies
        run: |
          pip install safety bandit semgrep
          pip install -r requirements.txt
          
      - name: Run safety check
        run: safety check --json --output safety_report.json
        
      - name: Run bandit security check
        run: bandit -r src/ -f json -o bandit_report.json
        
      - name: Run semgrep security scan
        run: semgrep --config=auto src/ --json -o semgrep_report.json
        
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety_report.json
            bandit_report.json
            semgrep_report.json
```

### Production Deployment Security

```bash
#!/bin/bash
# Production deployment script with security checks

echo "🔒 Starting secure deployment..."

# 1. Security validation
echo "Running security checks..."
safety check --file requirements.txt || exit 1
bandit -r src/ || exit 1

# 2. Environment validation
echo "Validating environment security..."
python scripts/validate_security_config.py || exit 1

# 3. Deploy with security
echo "Deploying with security hardening..."
docker build --security-opt no-new-privileges -t claude-tiu:secure .

# 4. Runtime security
echo "Enabling runtime security monitoring..."
kubectl apply -f k8s/security-policy.yaml
kubectl apply -f k8s/network-policy.yaml

echo "✅ Secure deployment complete!"
```

### Security Maintenance

#### Weekly Tasks
```bash
# Weekly security maintenance script
#!/bin/bash

echo "🔍 Weekly security maintenance..."

# Update dependencies
pip-review --auto
safety check --file requirements.txt

# Rotate API keys (if needed)
python scripts/rotate_api_keys.py

# Security scan
bandit -r src/ --format json -o weekly_security_scan.json

# Generate security report
python scripts/generate_security_report.py
```

#### Monthly Tasks
- Full penetration testing
- Security configuration review
- Access control audit
- Dependency vulnerability assessment
- Security training updates

### Security Metrics Dashboard

```python
# Security metrics collection
class SecurityMetrics:
    def collect_metrics(self):
        return {
            "threats_blocked": self.count_blocked_threats(),
            "authentication_failures": self.count_auth_failures(),
            "api_key_rotations": self.count_key_rotations(),
            "security_incidents": self.count_incidents(),
            "compliance_score": self.calculate_compliance_score()
        }
```

This guide provides a comprehensive implementation path for maintaining and enhancing the security posture of the Claude-TIU project.