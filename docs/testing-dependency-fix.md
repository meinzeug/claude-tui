# Critical Testing Dependencies Fix
## Immediate Action Plan für Test-Suite Restoration

**Status**: URGENT - Test Infrastructure Down  
**Impact**: Complete test suite non-functional (52 collection errors)  
**Priority**: CRITICAL - Blocking all quality assurance  

---

## 🚨 IMMEDIATE DEPENDENCY RESOLUTION

### Phase 1A: Core Dependencies (Execute Immediately)

```bash
# 1. Web Framework & API Testing
pip install fastapi[all]==0.104.1
pip install httpx==0.25.2  
pip install starlette==0.27.0
pip install uvicorn[standard]==0.24.0

# 2. Database & Async Support  
pip install sqlalchemy[postgresql,asyncio]==2.0.23
pip install asyncpg==0.29.0
pip install alembic==1.12.1
pip install databases[postgresql]==0.8.0

# 3. Redis & Caching
pip install redis==5.0.1
pip install aioredis==2.0.1

# 4. Scientific Computing & ML
pip install numpy==1.24.4
pip install pandas==2.1.3
pip install scikit-learn==1.3.2

# 5. Testing Framework Enhancement
pip install pytest-asyncio==0.21.1
pip install pytest-timeout==2.2.0
pip install pytest-xdist==3.3.1
pip install pytest-mock==3.12.0
```

### Phase 1B: Advanced Testing Dependencies

```bash
# Performance & Benchmarking
pip install pytest-benchmark==4.0.0
pip install memory-profiler==0.61.0
pip install psutil==5.9.6
pip install locust==2.17.0

# Property-Based & Data Generation
pip install hypothesis==6.92.1
pip install faker==20.1.0
pip install factory-boy==3.3.0

# Security Testing
pip install bandit==1.7.5
pip install safety==2.3.5
pip install cryptography==41.0.8

# UI & TUI Testing  
pip install textual[dev]==0.44.0
pip install rich==13.7.0

# Reporting & Visualization
pip install pytest-html==4.1.1
pip install pytest-json-report==1.5.0
pip install coverage[toml]==7.3.2
```

---

## 🔧 CRITICAL IMPORT FIXES

### Fix 1: FastAPI/Starlette TestClient Issue

**Fehlerhafte Module**: 52 Test-Dateien  
**Root Cause**: `httpx` dependency missing for `starlette.testclient`  

```python
# Before (BROKEN):
from fastapi.testclient import TestClient  # RuntimeError: httpx required

# After (WORKING):
pip install httpx  # First install dependency
from fastapi.testclient import TestClient  # Now works
```

### Fix 2: Redis Integration Issues

**Betroffen**: `src/ai/claude_flow_orchestrator.py`, alle AI-Tests  
**Root Cause**: `redis.asyncio` module missing  

```python
# Before (BROKEN):
import redis.asyncio as redis  # ModuleNotFoundError

# After (WORKING):  
pip install redis aioredis  # Install both sync and async redis
import redis.asyncio as redis  # Now works
```

### Fix 3: NumPy Analytics Issues

**Betroffen**: Alle Analytics-Tests (4 Dateien)  
**Root Cause**: Scientific computing stack missing  

```python
# Before (BROKEN):
import numpy as np  # ModuleNotFoundError  

# After (WORKING):
pip install numpy pandas scikit-learn  # Full data science stack
import numpy as np  # Now works
```

### Fix 4: Pydantic Version Conflicts

**Betroffen**: Community Platform Tests  
**Root Cause**: Pydantic v1 vs v2 compatibility issues  

```python
# Fix pydantic compatibility
pip install "pydantic>=2.5.0,<3.0.0"  # Ensure v2
pip install pydantic-settings==2.1.0   # Settings support
```

---

## ⚡ QUICK RESTORATION SCRIPT

**Execute this script immediately to restore basic test capability:**

```bash
#!/bin/bash
# quick-test-fix.sh - Immediate test restoration

echo "🚨 EMERGENCY TEST RESTORATION - Phase 1"

# Critical web dependencies
pip install fastapi[all]==0.104.1 httpx==0.25.2 starlette==0.27.0

# Critical database dependencies  
pip install sqlalchemy[postgresql,asyncio]==2.0.23 asyncpg==0.29.0

# Critical AI dependencies
pip install redis==5.0.1 aioredis==2.0.1

# Critical analytics dependencies
pip install numpy==1.24.4 pandas==2.1.3

# Critical async testing
pip install pytest-asyncio==0.21.1

echo "✅ Phase 1 dependencies installed"
echo "🧪 Testing basic functionality..."

# Test basic imports
python3 -c "
import fastapi
import httpx  
import redis.asyncio
import numpy
import pytest_asyncio
print('✅ All critical imports working')
"

echo "🚀 Running smoke test..."
python3 -m pytest tests/ -x --tb=no -q --disable-warnings 2>/dev/null && echo "✅ Tests can execute" || echo "❌ Additional fixes needed"

echo "📋 Next: Run full dependency installation script"
```

---

## 📋 FULL REQUIREMENTS.TXT UPDATE

**Add these to requirements-dev.txt:**

```txt
# Web Framework & API
fastapi[all]==0.104.1
httpx==0.25.2
starlette==0.27.0
uvicorn[standard]==0.24.0

# Database & ORM
sqlalchemy[postgresql,asyncio]==2.0.23
asyncpg==0.29.0
alembic==1.12.1
databases[postgresql]==0.8.0

# Redis & Caching  
redis==5.0.1
aioredis==2.0.1

# Scientific Computing
numpy==1.24.4
pandas==2.1.3
scikit-learn==1.3.2

# Testing Framework
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-timeout==2.2.0
pytest-xdist==3.3.1
pytest-benchmark==4.0.0
pytest-html==4.1.1
pytest-json-report==1.5.0

# Property-Based Testing
hypothesis==6.92.1
faker==20.1.0
factory-boy==3.3.0

# Security
bandit==1.7.5
safety==2.3.5
cryptography==41.0.8

# Performance
memory-profiler==0.61.0
psutil==5.9.6
locust==2.17.0

# UI/TUI
textual[dev]==0.44.0
rich==13.7.0

# Data Validation
pydantic>=2.5.0,<3.0.0
pydantic-settings==2.1.0
email-validator==2.1.0

# Development Tools
coverage[toml]==7.3.2
black==23.11.0
isort==5.12.0
mypy==1.7.1
```

---

## ⏰ EXECUTION TIMELINE

**IMMEDIATE (Next 30 minutes)**:
1. Execute quick-test-fix.sh script  
2. Verify basic test execution capability
3. Identify remaining critical blockers

**SHORT TERM (Next 2 hours)**:
1. Full dependency installation from requirements-dev.txt
2. Fix remaining import errors
3. Achieve basic test suite execution

**MEDIUM TERM (Next 24 hours)**:
1. Comprehensive test configuration validation
2. CI/CD pipeline restoration
3. Basic quality gates operational

---

## 🎯 SUCCESS CRITERIA

**Phase 1 Success**: 
- [ ] Zero collection errors in pytest
- [ ] Basic smoke tests passing  
- [ ] All critical imports working
- [ ] CI/CD pipeline can execute tests

**Phase 2 Success**:
- [ ] Full test suite executable
- [ ] Coverage reporting functional  
- [ ] Performance tests operational
- [ ] Security tests operational

---

**URGENT ACTION REQUIRED**: Execute dependency fixes immediately to restore testing capability.

*Test restoration critical for project quality assurance.*