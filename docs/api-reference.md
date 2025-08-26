# Claude-TIU API Reference

## Interactive API Documentation

**🌐 Try it Live:** [https://api.claude-tui.dev/docs](https://api.claude-tui.dev/docs)  
**📋 OpenAPI Spec:** [Download YAML](./openapi-specification.yaml)  
**🔗 Postman Collection:** [Download Collection](https://api.claude-tui.dev/postman)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
4. [Code Examples](#code-examples)
5. [Rate Limits](#rate-limits)
6. [Error Handling](#error-handling)
7. [WebSocket API](#websocket-api)
8. [SDKs and Libraries](#sdks-and-libraries)

---

## Getting Started

### Base URLs

| Environment | Base URL | Description |
|-------------|----------|-------------|
| Production | `https://api.claude-tui.dev/v1` | Production API server |
| Staging | `https://staging-api.claude-tui.dev/v1` | Staging environment |
| Local | `http://localhost:8000/api/v1` | Local development |

### Quick Test

```bash
# Health check (no authentication required)
curl -X GET "https://api.claude-tui.dev/v1/health"

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": 86400
}
```

---

## Authentication

### JWT Bearer Token Authentication

Claude-TIU uses JWT tokens for authentication. Obtain a token via the login endpoint:

```bash
# Login to get JWT token
curl -X POST "https://api.claude-tui.dev/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "developer@company.com",
    "password": "SecurePassword123!"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user-123",
    "email": "developer@company.com",
    "name": "John Developer",
    "role": "developer"
  }
}
```

### Using the Token

Include the token in the Authorization header for all API requests:

```bash
curl -X GET "https://api.claude-tui.dev/v1/projects" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### API Key Authentication (Service-to-Service)

For service-to-service communication, use API keys:

```bash
curl -X GET "https://api.claude-tui.dev/v1/projects" \
  -H "X-API-Key: your-api-key-here"
```

### Token Refresh

Refresh expired tokens using the refresh endpoint:

```bash
curl -X POST "https://api.claude-tui.dev/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token-here"
  }'
```

---

## Core Endpoints

### 🏗️ Project Management

#### Create New Project

Create an AI-powered development project with intelligent scaffolding:

**Endpoint:** `POST /projects`  
**Rate Limit:** 5 requests/minute

```bash
curl -X POST "https://api.claude-tui.dev/v1/projects" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Modern E-commerce API",
    "path": "/projects/ecommerce-api",
    "project_type": "fastapi",
    "template": "fastapi-advanced",
    "description": "High-performance e-commerce REST API with AI features",
    "features": [
      "authentication",
      "payment_processing", 
      "inventory_management",
      "ai_recommendations",
      "real_time_analytics"
    ],
    "initialize_git": true,
    "config": {
      "ai_creativity": 0.8,
      "validation_level": "strict",
      "auto_optimize": true,
      "database": "postgresql",
      "cache": "redis",
      "monitoring": "prometheus"
    }
  }'
```

**Response:**
```json
{
  "project_id": "proj-ecom-abc123",
  "name": "Modern E-commerce API",
  "type": "fastapi",
  "description": "High-performance e-commerce REST API with AI features",
  "path": "/projects/ecommerce-api",
  "created_at": "2024-01-15T10:30:00Z",
  "status": "active",
  "health_score": 0.95,
  "git_initialized": true,
  "config": {
    "ai_creativity": 0.8,
    "validation_level": "strict",
    "features_enabled": [
      "authentication",
      "payment_processing",
      "inventory_management"
    ],
    "estimated_completion": "2024-01-15T12:30:00Z"
  }
}
```

#### List Projects with Advanced Filtering

**Endpoint:** `GET /projects`

```bash
# Get all projects with pagination
curl -X GET "https://api.claude-tui.dev/v1/projects?page=1&page_size=20" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Filter by type and status
curl -X GET "https://api.claude-tui.dev/v1/projects?type=fastapi&status=active&page=1" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "projects": [
    {
      "project_id": "proj-ecom-abc123",
      "name": "Modern E-commerce API",
      "type": "fastapi",
      "description": "High-performance e-commerce REST API",
      "created_at": "2024-01-15T10:30:00Z",
      "status": "active",
      "health_score": 0.95,
      "recent_activity": "Code generation completed"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 1,
    "has_next": false,
    "has_prev": false
  }
}
```

#### Validate Project with Anti-Hallucination Engine

**Endpoint:** `POST /projects/{project_id}/validate`

```bash
curl -X POST "https://api.claude-tui.dev/v1/projects/proj-ecom-abc123/validate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "validation_level": "comprehensive",
    "include_tests": true,
    "check_security": true,
    "auto_fix": true
  }'
```

**Response:**
```json
{
  "project_id": "proj-ecom-abc123",
  "is_valid": true,
  "overall_score": 0.94,
  "authenticity_score": 0.96,
  "completeness_score": 0.92,
  "quality_score": 0.95,
  "issues": [
    {
      "file": "src/auth/models.py",
      "line": 45,
      "severity": "medium",
      "type": "placeholder",
      "description": "TODO comment found - incomplete implementation",
      "suggestion": "Complete the password validation logic",
      "auto_fixable": true
    }
  ],
  "warnings": [
    "Consider adding more comprehensive error handling in payment module"
  ],
  "recommendations": [
    "Add integration tests for payment processing",
    "Implement rate limiting for public endpoints",
    "Add API documentation with OpenAPI"
  ],
  "validated_at": "2024-01-15T10:35:00Z"
}
```

### 🤖 AI Integration

#### Generate Code with Claude Code

**Endpoint:** `POST /ai/code/generate`  
**Rate Limit:** 10 requests/minute

```bash
curl -X POST "https://api.claude-tui.dev/v1/ai/code/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a FastAPI endpoint for user authentication with JWT tokens, including registration, login, and password reset functionality",
    "language": "python",
    "context": {
      "framework": "fastapi",
      "database": "postgresql",
      "auth_library": "passlib",
      "include_tests": true,
      "security_level": "high"
    },
    "validate_response": true,
    "use_cache": true
  }'
```

**Response:**
```json
{
  "code": "from fastapi import APIRouter, Depends, HTTPException, status\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\nfrom pydantic import BaseModel, EmailStr\nfrom passlib.context import CryptContext\nfrom datetime import datetime, timedelta\nimport jwt\n\n# Authentication models\nclass UserRegistration(BaseModel):\n    email: EmailStr\n    password: str\n    name: str\n\n# Continue with full implementation...",
  "language": "python",
  "validation": {
    "is_valid": true,
    "quality_score": 0.93,
    "issues": [],
    "completeness": "fully_implemented",
    "security_score": 0.95
  },
  "metadata": {
    "lines_of_code": 187,
    "complexity_score": 0.7,
    "test_coverage": "85%",
    "security_features": [
      "password_hashing",
      "jwt_tokens",
      "input_validation",
      "rate_limiting"
    ]
  },
  "cached": false,
  "generated_at": "2024-01-15T10:40:00Z"
}
```

#### AI Code Review

**Endpoint:** `POST /ai/code/review`

```bash
curl -X POST "https://api.claude-tui.dev/v1/ai/code/review" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def process_payment(amount, card_number):\n    # TODO: Add validation\n    return {\"status\": \"success\"}",
    "language": "python",
    "review_criteria": {
      "check_security": true,
      "check_performance": true,
      "check_style": true,
      "check_error_handling": true
    }
  }'
```

**Response:**
```json
{
  "review": "# Code Review Report\n\n## Security Issues (Critical)\n- Card number is not validated or sanitized\n- No encryption for sensitive payment data\n- Missing authentication checks\n\n## Code Quality Issues\n- TODO comment indicates incomplete implementation\n- No error handling for payment failures\n- Function lacks proper type hints\n\n## Recommendations\n1. Implement proper PCI DSS compliance\n2. Add input validation for all parameters\n3. Use secure payment processing libraries\n4. Add comprehensive error handling\n5. Implement logging for audit trails",
  "overall_score": 0.25,
  "issues": [
    {
      "line": 1,
      "severity": "critical",
      "category": "security",
      "description": "Payment function lacks security validation",
      "suggestion": "Implement PCI DSS compliant payment processing"
    },
    {
      "line": 2,
      "severity": "high",
      "category": "logic",
      "description": "TODO comment indicates incomplete implementation", 
      "suggestion": "Complete the validation logic before deployment"
    }
  ],
  "recommendations": [
    "Use a secure payment processor like Stripe or PayPal",
    "Implement proper input validation and sanitization",
    "Add comprehensive error handling and logging",
    "Consider using type hints for better code clarity"
  ],
  "reviewed_at": "2024-01-15T10:45:00Z"
}
```

### ⚙️ Task Orchestration

#### Create Development Task

**Endpoint:** `POST /tasks`  
**Rate Limit:** 10 requests/minute

```bash
curl -X POST "https://api.claude-tui.dev/v1/tasks" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Implement User Management System",
    "description": "Create complete user management system with CRUD operations, authentication, and role-based access control",
    "task_type": "code_generation",
    "priority": "high",
    "timeout_seconds": 600,
    "dependencies": [],
    "ai_enabled": true,
    "config": {
      "language": "python",
      "framework": "fastapi",
      "features": [
        "user_crud",
        "authentication",
        "rbac",
        "password_reset",
        "email_verification"
      ],
      "include_tests": true,
      "include_docs": true,
      "database": "postgresql"
    }
  }'
```

**Response:**
```json
{
  "task_id": "task-user-mgmt-xyz789",
  "name": "Implement User Management System",
  "description": "Create complete user management system with CRUD operations",
  "task_type": "code_generation",
  "status": "pending",
  "priority": "high",
  "created_at": "2024-01-15T11:00:00Z",
  "estimated_duration": 450,
  "ai_enabled": true,
  "config": {
    "assigned_agents": ["coder", "tester", "reviewer"],
    "complexity_score": 0.75,
    "success_probability": 0.92
  }
}
```

#### Execute Task with Real-time Monitoring

**Endpoint:** `POST /tasks/{task_id}/execute`

```bash
curl -X POST "https://api.claude-tui.dev/v1/tasks/task-user-mgmt-xyz789/execute" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "execution_mode": "adaptive",
    "wait_for_dependencies": true,
    "enable_monitoring": true
  }'
```

**Response:**
```json
{
  "task_id": "task-user-mgmt-xyz789",
  "execution_id": "exec-abc123def456",
  "status": "started",
  "started_at": "2024-01-15T11:05:00Z",
  "estimated_completion": "2024-01-15T11:12:30Z",
  "progress": 0.0,
  "current_step": "analyzing_requirements",
  "agents_assigned": ["coder-001", "tester-002", "reviewer-003"],
  "monitoring_url": "wss://api.claude-tui.dev/v1/ws/tasks/task-user-mgmt-xyz789/status"
}
```

### 🔍 Advanced Validation

#### Run Anti-Hallucination Validation

**Endpoint:** `POST /validation/analyze`

```bash
curl -X POST "https://api.claude-tui.dev/v1/validation/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj-ecom-abc123",
    "validation_type": "comprehensive",
    "include_security_scan": true,
    "auto_fix_issues": true,
    "generate_report": true
  }'
```

**Response:**
```json
{
  "validation_id": "val-comprehensive-789",
  "project_id": "proj-ecom-abc123",
  "status": "completed",
  "overall_score": 0.91,
  "categories": {
    "authenticity": {
      "score": 0.94,
      "issues_found": 2,
      "placeholders_detected": 1,
      "incomplete_implementations": 1
    },
    "security": {
      "score": 0.89,
      "vulnerabilities": 3,
      "critical_issues": 0,
      "recommendations": 5
    },
    "quality": {
      "score": 0.92,
      "code_style_issues": 4,
      "performance_issues": 2,
      "maintainability_score": 0.88
    }
  },
  "auto_fixes_applied": 6,
  "manual_fixes_required": 3,
  "validated_at": "2024-01-15T11:15:00Z"
}
```

---

## Code Examples

### Python SDK Usage

```python
import asyncio
from claude_tui_sdk import ClaudeTIUClient, ProjectConfig, TaskConfig

async def main():
    # Initialize client
    client = ClaudeTIUClient(
        api_key="your-api-key",
        base_url="https://api.claude-tui.dev/v1"
    )
    
    # Create project
    project_config = ProjectConfig(
        name="AI-Powered Blog",
        project_type="fastapi",
        template="blog-cms",
        features=["authentication", "content_management", "ai_writing"]
    )
    
    project = await client.projects.create(project_config)
    print(f"Created project: {project.project_id}")
    
    # Generate code with AI
    code_result = await client.ai.generate_code(
        prompt="Create a blog post model with AI content generation",
        language="python",
        context={"framework": "fastapi", "orm": "sqlalchemy"}
    )
    
    print(f"Generated {len(code_result.code)} lines of code")
    
    # Validate project
    validation = await client.projects.validate(
        project.project_id,
        level="comprehensive"
    )
    
    print(f"Validation score: {validation.overall_score}")
    
    await client.close()

# Run async function
asyncio.run(main())
```

### JavaScript/TypeScript SDK

```typescript
import { ClaudeTIUClient, ProjectType } from '@claude-tui/sdk';

const client = new ClaudeTIUClient({
  apiKey: process.env.CLAUDE_TIU_API_KEY,
  baseUrl: 'https://api.claude-tui.dev/v1'
});

async function createReactApp() {
  try {
    // Create React TypeScript project
    const project = await client.projects.create({
      name: 'Modern React Dashboard',
      projectType: ProjectType.REACT,
      template: 'react-typescript-dashboard',
      features: ['routing', 'state-management', 'charts', 'authentication'],
      config: {
        aiCreativity: 0.8,
        validationLevel: 'strict',
        autoOptimize: true
      }
    });

    console.log(`Project created: ${project.projectId}`);

    // Generate React component with AI
    const componentCode = await client.ai.generateCode({
      prompt: 'Create a reusable data table component with sorting, filtering, and pagination',
      language: 'typescript',
      context: {
        framework: 'react',
        styling: 'tailwindcss',
        includeTests: true
      }
    });

    console.log('Generated React component:', componentCode.metadata);

    // Validate the generated code
    const validation = await client.validation.analyzeCode({
      code: componentCode.code,
      language: 'typescript',
      validationType: 'react_component'
    });

    console.log(`Code quality score: ${validation.qualityScore}`);
    
  } catch (error) {
    console.error('Error:', error);
  }
}

createReactApp();
```

### cURL Examples Collection

**Authentication Flow:**
```bash
#!/bin/bash

# Set base URL
BASE_URL="https://api.claude-tui.dev/v1"

# 1. Register new user
curl -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "developer@company.com",
    "password": "SecurePassword123!",
    "name": "Jane Developer",
    "company": "Tech Corp"
  }'

# 2. Login and save token
TOKEN=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "developer@company.com", 
    "password": "SecurePassword123!"
  }' | jq -r '.access_token')

echo "Token: $TOKEN"

# 3. Create project using token
curl -X POST "$BASE_URL/projects" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Microservices API",
    "project_type": "fastapi",
    "template": "microservices-template"
  }'
```

**Workflow Automation:**
```bash
#!/bin/bash

# Create and execute a complete development workflow
TOKEN="your-jwt-token-here"
BASE_URL="https://api.claude-tui.dev/v1"

# 1. Create workflow
WORKFLOW_ID=$(curl -s -X POST "$BASE_URL/workflows" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Full Stack Development Workflow",
    "description": "Complete development pipeline from design to deployment",
    "steps": [
      {
        "name": "API Development",
        "type": "code_generation",
        "config": {
          "language": "python",
          "framework": "fastapi",
          "features": ["authentication", "crud_operations", "websockets"]
        }
      },
      {
        "name": "Frontend Development", 
        "type": "code_generation",
        "dependencies": ["API Development"],
        "config": {
          "language": "typescript",
          "framework": "react",
          "features": ["dashboard", "forms", "real_time_updates"]
        }
      },
      {
        "name": "Testing Suite",
        "type": "testing",
        "dependencies": ["API Development", "Frontend Development"],
        "config": {
          "test_types": ["unit", "integration", "e2e"],
          "coverage_threshold": 0.85
        }
      },
      {
        "name": "Deployment Setup",
        "type": "deployment",
        "dependencies": ["Testing Suite"],
        "config": {
          "platform": "kubernetes",
          "environment": "production"
        }
      }
    ]
  }' | jq -r '.workflow_id')

echo "Created workflow: $WORKFLOW_ID"

# 2. Execute workflow
EXECUTION_ID=$(curl -s -X POST "$BASE_URL/workflows/$WORKFLOW_ID/execute" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "adaptive",
    "variables": {
      "project_name": "Enterprise Dashboard",
      "database": "postgresql",
      "cache": "redis"
    }
  }' | jq -r '.execution_id')

echo "Executing workflow: $EXECUTION_ID"

# 3. Monitor progress
while true; do
  STATUS=$(curl -s -X GET "$BASE_URL/workflows/$WORKFLOW_ID/executions/$EXECUTION_ID" \
    -H "Authorization: Bearer $TOKEN" | jq -r '.status')
  
  if [ "$STATUS" == "completed" ] || [ "$STATUS" == "failed" ]; then
    echo "Workflow $STATUS"
    break
  fi
  
  echo "Status: $STATUS"
  sleep 5
done
```

---

## Rate Limits

Claude-TIU implements intelligent rate limiting to ensure fair usage and optimal performance:

| Endpoint Category | Rate Limit | Window | Burst Limit |
|-------------------|------------|--------|-------------|
| Authentication | 10 requests | 1 minute | 20 |
| Project Operations | 20 requests | 1 minute | 50 |
| AI Code Generation | 10 requests | 1 minute | 15 |
| Task Execution | 5 requests | 1 minute | 10 |
| Validation | 15 requests | 1 minute | 30 |
| Analytics | 30 requests | 1 minute | 60 |
| WebSocket Connections | 5 connections | 1 minute | 10 |

### Rate Limit Headers

Every API response includes rate limit information:

```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1642248600
X-RateLimit-Burst-Remaining: 40
```

### Handling Rate Limits

```python
import time
import requests

def make_api_request(url, headers, data=None):
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 429:
        # Rate limit exceeded
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        wait_time = reset_time - int(time.time())
        
        if wait_time > 0:
            print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return make_api_request(url, headers, data)  # Retry
    
    return response
```

### Enterprise Rate Limits

Enterprise customers get higher rate limits:

- **Standard:** 2x base limits
- **Professional:** 5x base limits  
- **Enterprise:** 10x base limits + priority queuing

---

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_ERROR_CODE",
  "status_code": 400,
  "details": {
    "field": "validation_errors",
    "context": "additional_information"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req-abc123def456"
}
```

### Common Error Codes

| Status Code | Error Code | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | `INVALID_INPUT` | Request validation failed | Check request format and required fields |
| 401 | `UNAUTHORIZED` | Authentication required | Include valid JWT token in Authorization header |
| 403 | `FORBIDDEN` | Insufficient permissions | Check user role and permissions |
| 404 | `NOT_FOUND` | Resource doesn't exist | Verify resource ID and existence |
| 409 | `CONFLICT` | Resource already exists | Use different name or update existing resource |
| 422 | `VALIDATION_ERROR` | Input validation failed | Fix validation errors listed in response |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry after rate limit resets |
| 500 | `INTERNAL_ERROR` | Server error | Contact support with request ID |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily down | Check system status and retry later |

### Error Handling Best Practices

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def handle_api_errors(response):
    if response.status_code == 200:
        return response.json()
    
    error_data = response.json()
    error_code = error_data.get('error_code')
    
    if error_code == 'RATE_LIMIT_EXCEEDED':
        retry_after = error_data.get('retry_after', 60)
        raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")
    
    elif error_code == 'VALIDATION_ERROR':
        validation_errors = error_data.get('validation_errors', [])
        raise ValidationError("Input validation failed", validation_errors)
    
    elif error_code == 'UNAUTHORIZED':
        raise AuthenticationError("Authentication required or token expired")
    
    else:
        raise APIError(f"API Error: {error_data.get('error')}", error_code)

# Usage example
try:
    session = create_session_with_retries()
    response = session.post(
        "https://api.claude-tui.dev/v1/projects",
        headers={"Authorization": f"Bearer {token}"},
        json=project_data
    )
    result = handle_api_errors(response)
    print("Project created successfully:", result)
    
except RateLimitError as e:
    print(f"Rate limit error: {e}")
    # Implement backoff strategy
    
except ValidationError as e:
    print(f"Validation error: {e.errors}")
    # Fix input data and retry
    
except AuthenticationError as e:
    print(f"Auth error: {e}")
    # Refresh token and retry
```

---

## WebSocket API

### Real-time Project Progress

Connect to receive live updates about project creation and modification:

```javascript
const ws = new WebSocket(
  'wss://api.claude-tui.dev/v1/ws/projects/proj-abc123/progress',
  ['token', 'your-jwt-token-here']
);

ws.onopen = function() {
  console.log('Connected to project progress stream');
};

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Progress update:', update);
  
  switch(update.type) {
    case 'progress':
      updateProgressBar(update.percentage);
      break;
    case 'step_completed':
      logCompletedStep(update.step_name, update.duration);
      break;
    case 'validation_result':
      displayValidationResults(update.results);
      break;
    case 'error':
      handleError(update.error);
      break;
  }
};
```

### Task Execution Monitoring

```javascript
const taskWs = new WebSocket(
  'wss://api.claude-tui.dev/v1/ws/tasks/task-xyz789/status',
  ['token', 'your-jwt-token-here']
);

taskWs.onmessage = function(event) {
  const status = JSON.parse(event.data);
  
  switch(status.event) {
    case 'task_started':
      console.log(`Task started with ${status.agents.length} agents`);
      break;
      
    case 'agent_progress':
      console.log(`Agent ${status.agent_id}: ${status.progress}%`);
      break;
      
    case 'code_generated':
      displayGeneratedCode(status.code, status.language);
      break;
      
    case 'validation_complete':
      showValidationResults(status.validation_report);
      break;
      
    case 'task_completed':
      console.log(`Task completed in ${status.duration}s`);
      downloadResults(status.result_urls);
      break;
  }
};
```

### AI Agent Coordination Stream

```javascript
const coordinationWs = new WebSocket(
  'wss://api.claude-tui.dev/v1/ws/workflows/workflow-123/coordination'
);

coordinationWs.onmessage = function(event) {
  const coord = JSON.parse(event.data);
  
  if (coord.type === 'agent_communication') {
    console.log(`${coord.from_agent} → ${coord.to_agent}: ${coord.message}`);
  }
  
  if (coord.type === 'swarm_decision') {
    console.log(`Swarm decided: ${coord.decision} (confidence: ${coord.confidence})`);
  }
};
```

---

## SDKs and Libraries

### Official SDKs

| Language | Package | Installation | Documentation |
|----------|---------|-------------|---------------|
| Python | `claude-tui-sdk` | `pip install claude-tui-sdk` | [Python Docs](https://docs.claude-tui.dev/sdk/python) |
| JavaScript/TypeScript | `@claude-tui/sdk` | `npm install @claude-tui/sdk` | [JS/TS Docs](https://docs.claude-tui.dev/sdk/javascript) |
| Go | `github.com/claude-tui/go-sdk` | `go get github.com/claude-tui/go-sdk` | [Go Docs](https://docs.claude-tui.dev/sdk/go) |
| Java | `com.claudetiu:claude-tui-sdk` | Maven/Gradle | [Java Docs](https://docs.claude-tui.dev/sdk/java) |

### Community SDKs

| Language | Repository | Maintainer | Status |
|----------|------------|------------|--------|
| Ruby | [claude-tui-ruby](https://github.com/community/claude-tui-ruby) | @rubydev | Active |
| PHP | [claude-tui-php](https://github.com/community/claude-tui-php) | @phpmaster | Active |
| C# | [ClaudeTIU.NET](https://github.com/community/claude-tui-dotnet) | @dotnetguru | Beta |
| Rust | [claude-tui-rs](https://github.com/community/claude-tui-rust) | @rustacean | Alpha |

### CLI Tool

```bash
# Install CLI
npm install -g @claude-tui/cli

# Configure
claude-tui config set api-key YOUR_API_KEY
claude-tui config set base-url https://api.claude-tui.dev/v1

# Create project
claude-tui projects create \
  --name "My API Project" \
  --type fastapi \
  --template advanced-api \
  --features auth,database,testing

# Execute task
claude-tui tasks create \
  --name "Implement user management" \
  --type code_generation \
  --config features=crud,auth,validation

# Monitor progress
claude-tui tasks watch task-abc123
```

### Postman Collection

Import our comprehensive Postman collection for easy API testing:

**Import URL:** `https://api.claude-tui.dev/postman/collection.json`

The collection includes:
- All API endpoints with example requests
- Environment variables for different stages
- Pre-request scripts for authentication
- Test scripts for response validation
- Documentation for each endpoint

### OpenAPI Integration

Generate client SDKs from our OpenAPI specification:

```bash
# Generate Python client
openapi-generator generate \
  -i https://api.claude-tui.dev/v1/openapi.json \
  -g python \
  -o ./claude-tui-python-client

# Generate JavaScript client  
openapi-generator generate \
  -i https://api.claude-tui.dev/v1/openapi.json \
  -g javascript \
  -o ./claude-tui-js-client
```

---

## Support and Resources

### Documentation Links

- **🏠 Main Documentation:** [https://docs.claude-tui.dev](https://docs.claude-tui.dev)
- **📖 API Reference:** [https://api.claude-tui.dev/docs](https://api.claude-tui.dev/docs)
- **🔧 SDK Documentation:** [https://docs.claude-tui.dev/sdks](https://docs.claude-tui.dev/sdks)
- **📊 Status Page:** [https://status.claude-tui.dev](https://status.claude-tui.dev)

### Support Channels

- **📧 Email Support:** support@claude-tui.dev
- **💬 Discord Community:** [https://discord.gg/claude-tui](https://discord.gg/claude-tui)  
- **🐛 Issue Tracker:** [https://github.com/claude-tui/issues](https://github.com/claude-tui/issues)
- **📚 Knowledge Base:** [https://help.claude-tui.dev](https://help.claude-tui.dev)

### Rate Limiting & Fair Use

We implement intelligent rate limiting to ensure optimal performance for all users. Enterprise customers receive higher limits and priority support.

---

**Happy Coding with Claude-TIU! 🚀**

*This API reference is automatically updated with each release. For the latest version, visit our [documentation site](https://docs.claude-tui.dev).*