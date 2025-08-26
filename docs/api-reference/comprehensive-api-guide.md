# Comprehensive API Reference

The Claude-TUI API provides comprehensive REST endpoints for integrating with the intelligent development brain. This reference covers all available endpoints, authentication methods, and integration patterns.

## 🌟 API Overview

### Base URL
```
Production: https://api.claude-tui.dev/v1
Staging:    https://staging-api.claude-tui.dev/v1
Local:      http://localhost:8000/api/v1
```

### Core Features

- **Project Management**: Complete CRUD operations for AI-powered projects
- **Task Orchestration**: Execute complex development workflows
- **AI Integration**: Access to 54+ specialized AI agents
- **Real-time Communication**: WebSocket support for live updates
- **Validation Engine**: Anti-hallucination quality assurance
- **Analytics**: Performance tracking and usage insights
- **Community**: Template marketplace and collaboration

### Response Standards

All API responses follow a consistent structure:

```json
{
  "success": true,
  "data": { /* response payload */ },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789",
  "meta": {
    "version": "1.0.0",
    "rate_limit": {
      "remaining": 98,
      "reset_at": "2024-01-15T11:00:00Z"
    }
  }
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid project configuration",
    "details": {
      "field": "name",
      "reason": "Project name cannot be empty"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## 🔐 Authentication

### JWT Token Authentication

The primary authentication method uses JWT tokens:

#### Login Endpoint

```http
POST /auth/login
Content-Type: application/json

{
  "email": "developer@company.com",
  "password": "SecurePass123!",
  "remember_me": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
      "id": "user_123",
      "email": "developer@company.com",
      "name": "John Developer",
      "role": "developer"
    }
  }
}
```

#### Using the Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

#### Refresh Token

```http
POST /auth/refresh
Authorization: Bearer your-refresh-token

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### API Key Authentication

For programmatic access, use API keys:

```http
X-API-Key: api_key_123456789abcdef
```

Create API keys through the dashboard or using the endpoint:

```http
POST /auth/api-keys
Authorization: Bearer your-jwt-token

{
  "name": "Production Integration",
  "permissions": ["projects:read", "projects:write", "tasks:execute"],
  "expires_at": "2025-01-15T00:00:00Z"
}
```

## 🚀 Project Management

### Create Project

Create a new AI-powered development project:

```http
POST /projects
Authorization: Bearer your-token
Content-Type: application/json

{
  "name": "my-awesome-app",
  "description": "A revolutionary web application",
  "template": "fullstack-react-fastapi",
  "configuration": {
    "language": "python",
    "frontend_framework": "react",
    "backend_framework": "fastapi",
    "database": "postgresql",
    "ai_assistance_level": "high",
    "testing_strategy": "comprehensive"
  },
  "requirements": {
    "features": [
      "user_authentication",
      "api_endpoints",
      "database_integration",
      "real_time_updates"
    ],
    "constraints": {
      "performance_targets": {
        "response_time": "< 200ms",
        "concurrent_users": 1000
      }
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "proj_123456",
    "name": "my-awesome-app",
    "status": "creating",
    "configuration": { /* ... */ },
    "creation_progress": {
      "current_step": "analyzing_requirements",
      "completion_percentage": 15,
      "estimated_completion": "2024-01-15T10:45:00Z"
    },
    "ai_agents": {
      "assigned": ["backend-dev", "frontend-dev", "test-engineer"],
      "active": 1,
      "queued": 2
    },
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

### List Projects

```http
GET /projects?page=1&limit=20&status=active&sort=created_at:desc
Authorization: Bearer your-token
```

### Get Project Details

```http
GET /projects/{project_id}
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "proj_123456",
    "name": "my-awesome-app",
    "status": "active",
    "structure": {
      "directories": ["src/", "tests/", "docs/"],
      "files": [
        {
          "path": "src/main.py",
          "type": "python",
          "size": 2048,
          "ai_generated": true,
          "validation_score": 0.98
        }
      ]
    },
    "metrics": {
      "total_files": 24,
      "lines_of_code": 3420,
      "test_coverage": 87.5,
      "validation_score": 0.96,
      "ai_contribution": 78.2
    },
    "recent_activity": [
      {
        "type": "file_created",
        "file": "src/api/auth.py",
        "agent": "backend-dev",
        "timestamp": "2024-01-15T10:25:00Z"
      }
    ]
  }
}
```

### Update Project

```http
PUT /projects/{project_id}
Authorization: Bearer your-token
Content-Type: application/json

{
  "description": "Updated description",
  "configuration": {
    "ai_assistance_level": "maximum"
  }
}
```

### Delete Project

```http
DELETE /projects/{project_id}
Authorization: Bearer your-token
```

## 🎯 Task Orchestration

### Execute Task

Execute a development task with AI assistance:

```http
POST /tasks/execute
Authorization: Bearer your-token
Content-Type: application/json

{
  "project_id": "proj_123456",
  "task": {
    "type": "feature_implementation",
    "title": "Implement User Authentication",
    "description": "Create complete user authentication system with JWT tokens",
    "requirements": {
      "components": ["login", "register", "password_reset"],
      "security_level": "high",
      "testing": "required"
    },
    "context": {
      "existing_files": ["src/models/user.py"],
      "dependencies": ["fastapi", "jwt", "bcrypt"]
    }
  },
  "execution_options": {
    "ai_agents": ["backend-dev", "security-auditor", "test-engineer"],
    "validation_level": "strict",
    "auto_fix_issues": true,
    "parallel_execution": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "task_789012",
    "status": "executing",
    "assigned_agents": [
      {
        "id": "backend-dev",
        "name": "Backend Developer",
        "status": "active",
        "current_subtask": "creating_auth_endpoints"
      }
    ],
    "progress": {
      "completion_percentage": 0,
      "current_phase": "analysis",
      "estimated_duration": "15 minutes",
      "subtasks": [
        {
          "id": "subtask_1",
          "title": "Analyze requirements",
          "status": "in_progress"
        }
      ]
    },
    "execution_plan": {
      "phases": ["analysis", "implementation", "testing", "validation"],
      "parallel_tracks": 2,
      "dependency_graph": { /* ... */ }
    }
  }
}
```

### Get Task Status

```http
GET /tasks/{task_id}/status
Authorization: Bearer your-token
```

### List Tasks

```http
GET /projects/{project_id}/tasks?status=active&page=1&limit=20
Authorization: Bearer your-token
```

### Cancel Task

```http
POST /tasks/{task_id}/cancel
Authorization: Bearer your-token
```

## 🧠 AI Agent Management

### List Available Agents

```http
GET /ai/agents
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "id": "backend-dev",
        "name": "Backend Developer",
        "description": "Specializes in server-side development and APIs",
        "capabilities": [
          "api_development",
          "database_integration",
          "authentication_systems",
          "performance_optimization"
        ],
        "supported_languages": ["python", "javascript", "go", "rust"],
        "frameworks": ["fastapi", "express", "gin", "actix"],
        "status": "available",
        "current_load": 0.3,
        "success_rate": 0.94
      }
    ],
    "total": 54,
    "available": 52,
    "busy": 2
  }
}
```

### Spawn Agent

```http
POST /ai/agents/spawn
Authorization: Bearer your-token
Content-Type: application/json

{
  "agent_type": "frontend-dev",
  "task_description": "Create a responsive React dashboard with real-time updates",
  "project_context": {
    "project_id": "proj_123456",
    "existing_components": ["Header", "Sidebar"],
    "design_system": "material-ui"
  },
  "configuration": {
    "creativity_level": "high",
    "code_style": "functional",
    "testing_approach": "jest"
  }
}
```

### Get Agent Status

```http
GET /ai/agents/{agent_id}/status
Authorization: Bearer your-token
```

### Agent Performance Metrics

```http
GET /ai/agents/{agent_id}/metrics?period=7d
Authorization: Bearer your-token
```

## 🔍 Validation & Quality Assurance

### Validate Code

Use the anti-hallucination engine to validate code:

```http
POST /validation/validate-code
Authorization: Bearer your-token
Content-Type: application/json

{
  "code": "def authenticate_user(email: str, password: str):\n    # TODO: implement authentication logic\n    pass",
  "language": "python",
  "context": {
    "project_id": "proj_123456",
    "file_path": "src/auth.py",
    "related_files": ["src/models/user.py"]
  },
  "validation_options": {
    "check_placeholders": true,
    "semantic_analysis": true,
    "security_scan": true,
    "auto_fix": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": false,
    "authenticity_score": 0.45,
    "issues": [
      {
        "type": "placeholder",
        "severity": "high",
        "line_number": 2,
        "description": "TODO comment indicates incomplete implementation",
        "suggestion": "Implement authentication logic with proper password hashing and validation"
      }
    ],
    "auto_fix_result": {
      "fixed_code": "def authenticate_user(email: str, password: str):\n    user = get_user_by_email(email)\n    if user and verify_password(password, user.password_hash):\n        return user\n    return None",
      "changes_made": ["Implemented authentication logic", "Added password verification"],
      "new_authenticity_score": 0.92
    },
    "recommendations": [
      "Add input validation for email format",
      "Implement rate limiting for authentication attempts"
    ]
  }
}
```

### Validate Project

```http
POST /validation/validate-project/{project_id}
Authorization: Bearer your-token
```

### Get Validation History

```http
GET /validation/history?project_id=proj_123456&limit=50
Authorization: Bearer your-token
```

## 📊 Analytics & Monitoring

### Project Analytics

```http
GET /analytics/projects/{project_id}?period=30d&metrics=all
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "project_id": "proj_123456",
    "period": {
      "start": "2023-12-16T00:00:00Z",
      "end": "2024-01-15T00:00:00Z"
    },
    "metrics": {
      "development_velocity": {
        "lines_per_day": 450.2,
        "features_per_week": 3.5,
        "bugs_fixed_per_day": 2.1
      },
      "ai_contribution": {
        "percentage": 78.5,
        "agent_utilization": {
          "backend-dev": 45.2,
          "frontend-dev": 32.1,
          "test-engineer": 22.7
        }
      },
      "quality_metrics": {
        "validation_score": 0.96,
        "test_coverage": 87.3,
        "security_score": 0.94
      },
      "performance": {
        "build_time": "2m 34s",
        "test_execution_time": "1m 12s",
        "deployment_time": "45s"
      }
    },
    "trends": {
      "velocity_trend": "increasing",
      "quality_trend": "stable",
      "agent_efficiency": "improving"
    }
  }
}
```

### System Metrics

```http
GET /analytics/system/metrics
Authorization: Bearer your-token
```

### Usage Analytics

```http
GET /analytics/usage?user_id=user_123&period=7d
Authorization: Bearer your-token
```

## 🌐 Real-time Communication

### WebSocket Connection

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://api.claude-tui.dev/v1/ws');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-jwt-token'
    }));
};

// Subscribe to project updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['project:proj_123456', 'tasks:*']
}));

// Handle messages
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

### Message Types

#### Task Progress Updates

```json
{
  "type": "task_progress",
  "task_id": "task_789012",
  "progress": {
    "completion_percentage": 65,
    "current_phase": "implementation",
    "agent_status": {
      "backend-dev": "active",
      "test-engineer": "queued"
    }
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

#### Validation Results

```json
{
  "type": "validation_complete",
  "project_id": "proj_123456",
  "file": "src/api/auth.py",
  "result": {
    "is_valid": true,
    "authenticity_score": 0.94,
    "issues_found": 0
  },
  "timestamp": "2024-01-15T10:36:00Z"
}
```

#### Agent Notifications

```json
{
  "type": "agent_notification",
  "agent_id": "backend-dev",
  "message": "Code generation complete for user authentication module",
  "severity": "info",
  "actions": [
    {
      "type": "review_code",
      "url": "/projects/proj_123456/files/src/auth.py"
    }
  ],
  "timestamp": "2024-01-15T10:37:00Z"
}
```

## 🏪 Community & Marketplace

### Browse Templates

```http
GET /community/templates?category=web&language=python&sort=popular
Authorization: Bearer your-token
```

### Download Template

```http
GET /community/templates/{template_id}/download
Authorization: Bearer your-token
```

### Upload Template

```http
POST /community/templates
Authorization: Bearer your-token
Content-Type: multipart/form-data

{
  "name": "FastAPI Microservice Template",
  "description": "Production-ready FastAPI microservice with authentication",
  "category": "backend",
  "tags": ["fastapi", "microservice", "authentication"],
  "template_file": [binary data],
  "preview_images": [image files]
}
```

### Rate Template

```http
POST /community/templates/{template_id}/rating
Authorization: Bearer your-token
Content-Type: application/json

{
  "rating": 5,
  "review": "Excellent template, saved me hours of setup time!"
}
```

## 🛠 Configuration & Settings

### Get User Settings

```http
GET /settings/user
Authorization: Bearer your-token
```

### Update Settings

```http
PUT /settings/user
Authorization: Bearer your-token
Content-Type: application/json

{
  "preferences": {
    "default_ai_assistance_level": "high",
    "code_style": "pep8",
    "notification_settings": {
      "task_completion": true,
      "validation_results": true,
      "agent_notifications": false
    }
  },
  "ai_configuration": {
    "preferred_agents": ["backend-dev", "test-engineer"],
    "creativity_level": "balanced",
    "validation_strictness": "high"
  }
}
```

## 🚨 Error Handling

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | Invalid credentials | 401 |
| `INSUFFICIENT_PERMISSIONS` | Access denied | 403 |
| `RESOURCE_NOT_FOUND` | Resource doesn't exist | 404 |
| `VALIDATION_ERROR` | Invalid input data | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `AI_SERVICE_UNAVAILABLE` | AI service is down | 503 |
| `TASK_EXECUTION_FAILED` | Task failed to execute | 422 |
| `QUOTA_EXCEEDED` | Usage quota exceeded | 402 |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid project configuration",
    "details": {
      "field": "configuration.language",
      "value": "unknown_language",
      "allowed_values": ["python", "javascript", "typescript", "go"]
    },
    "suggestion": "Use one of the supported programming languages"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789",
  "documentation_url": "https://docs.claude-tui.dev/api/errors#validation-error"
}
```

## 📈 Rate Limits

### Default Limits

| Endpoint Category | Requests per Hour | Burst Limit |
|------------------|-------------------|-------------|
| Authentication | 100 | 10 |
| Project Operations | 1000 | 50 |
| Task Execution | 500 | 20 |
| AI Agent Operations | 2000 | 100 |
| Analytics | 300 | 30 |
| Community | 500 | 25 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1642248000
X-RateLimit-Retry-After: 3600
```

## 🔧 SDKs & Libraries

### Python SDK

```python
from claude_tui import ClaudeTUIClient

client = ClaudeTUIClient(
    api_key="your-api-key",
    base_url="https://api.claude-tui.dev/v1"
)

# Create project
project = await client.projects.create({
    "name": "my-project",
    "template": "fastapi-service"
})

# Execute task
task = await client.tasks.execute({
    "project_id": project.id,
    "task": {
        "type": "feature_implementation",
        "description": "Add user authentication"
    }
})

# Monitor progress
async for progress in client.tasks.monitor(task.id):
    print(f"Progress: {progress.completion_percentage}%")
```

### JavaScript SDK

```javascript
import { ClaudeTUIClient } from '@claude-tui/client';

const client = new ClaudeTUIClient({
    apiKey: 'your-api-key',
    baseURL: 'https://api.claude-tui.dev/v1'
});

// Create project
const project = await client.projects.create({
    name: 'my-project',
    template: 'react-app'
});

// Execute task with real-time updates
const task = await client.tasks.execute({
    projectId: project.id,
    task: {
        type: 'component_creation',
        description: 'Create a user dashboard component'
    }
});

// Listen for updates
client.tasks.onProgress(task.id, (progress) => {
    console.log(`Task ${progress.completion_percentage}% complete`);
});
```

## 📚 Best Practices

### API Integration

1. **Authentication**: Always use HTTPS and secure token storage
2. **Error Handling**: Implement comprehensive error handling
3. **Rate Limiting**: Respect rate limits and implement backoff
4. **Monitoring**: Track API usage and performance
5. **Caching**: Cache responses when appropriate

### Task Execution

1. **Context**: Provide rich context for better AI performance
2. **Validation**: Always validate AI-generated code
3. **Monitoring**: Monitor task progress in real-time
4. **Cancellation**: Implement task cancellation for long operations
5. **Recovery**: Handle task failures gracefully

### Performance Optimization

1. **Pagination**: Use pagination for large datasets
2. **Filtering**: Apply filters to reduce response sizes
3. **Compression**: Enable gzip compression
4. **Connection Pooling**: Reuse HTTP connections
5. **Async Operations**: Use async operations for better throughput

---

*This comprehensive API reference provides everything you need to integrate with Claude-TUI's intelligent development brain. For more examples and advanced usage patterns, see our [SDK documentation](../sdks/) and [integration guides](../integrations/).*