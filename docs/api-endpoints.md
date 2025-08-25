# Claude TIU REST API Endpoints

## Overview

The Claude TIU REST API provides comprehensive endpoints for AI-powered development operations, including project management, task orchestration, validation services, and AI integration.

## Base URL

```
https://api.claude-tiu.example.com/api/v1
```

## Authentication

All API endpoints require JWT authentication via Bearer token:

```
Authorization: Bearer <your_jwt_token>
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- Standard endpoints: 60 requests/minute
- Heavy operations: 10 requests/minute  
- Batch operations: 3 requests/minute

## API Endpoints

### 1. Project Management (`/projects`)

#### Create Project
```http
POST /api/v1/projects
```

**Request Body:**
```json
{
  "name": "my-project",
  "path": "/path/to/project",
  "project_type": "python",
  "template": "fastapi",
  "description": "My AI-powered project",
  "initialize_git": true,
  "create_venv": true,
  "config": {}
}
```

**Response:**
```json
{
  "project_id": "uuid-here",
  "name": "my-project",
  "path": "/path/to/project",
  "project_type": "python",
  "status": "created",
  "created_at": "2025-01-01T00:00:00Z",
  "git_initialized": true,
  "venv_created": true
}
```

#### List Projects
```http
GET /api/v1/projects?page=1&page_size=10
```

#### Get Project Details
```http
GET /api/v1/projects/{project_id}
```

#### Validate Project
```http
POST /api/v1/projects/{project_id}/validate
```

#### Update Project Configuration
```http
PUT /api/v1/projects/{project_id}/config
```

#### Load Existing Project
```http
POST /api/v1/projects/{project_id}/load
```

#### Remove Project
```http
DELETE /api/v1/projects/{project_id}?delete_files=false
```

#### Project Health Check
```http
GET /api/v1/projects/{project_id}/health
```

### 2. Task Management (`/tasks`)

#### Create Task
```http
POST /api/v1/tasks
```

**Request Body:**
```json
{
  "name": "Generate API endpoints",
  "description": "Create REST API endpoints for user management",
  "task_type": "code_generation",
  "priority": "high",
  "timeout_seconds": 300,
  "dependencies": [],
  "ai_enabled": true,
  "config": {
    "language": "python",
    "framework": "fastapi"
  }
}
```

#### List Tasks
```http
GET /api/v1/tasks?page=1&page_size=10&task_type_filter=code_generation&status_filter=completed
```

#### Get Task Details
```http
GET /api/v1/tasks/{task_id}
```

#### Execute Task
```http
POST /api/v1/tasks/{task_id}/execute
```

**Request Body:**
```json
{
  "execution_mode": "adaptive",
  "wait_for_dependencies": true
}
```

#### Cancel Task
```http
POST /api/v1/tasks/{task_id}/cancel
```

#### Get Execution History
```http
GET /api/v1/tasks/history/execution?limit=50&task_type_filter=code_generation&success_only=true
```

#### Get Performance Report
```http
GET /api/v1/tasks/performance/report
```

#### Batch Create Tasks
```http
POST /api/v1/tasks/batch/create
```

#### Task Service Health
```http
GET /api/v1/tasks/health/service
```

### 3. Validation Services (`/validation`)

#### Validate Code
```http
POST /api/v1/validation/code
```

**Request Body:**
```json
{
  "code": "def hello_world():\n    print('Hello, World!')",
  "language": "python",
  "file_path": "hello.py",
  "validation_level": "standard",
  "check_placeholders": true,
  "check_syntax": true,
  "check_quality": true
}
```

**Response:**
```json
{
  "is_valid": true,
  "score": 0.95,
  "issues": [],
  "warnings": ["Consider adding docstring"],
  "suggestions": ["Add type hints"],
  "categories": {
    "syntax": {"is_valid": true},
    "quality": {"score": 0.9},
    "placeholder": {"count": 0}
  },
  "metadata": {
    "language": "python",
    "validated_at": "2025-01-01T00:00:00Z"
  }
}
```

#### Validate AI Response
```http
POST /api/v1/validation/response
```

#### Validate Progress Authenticity
```http
POST /api/v1/validation/progress
```

#### Validate File Upload
```http
POST /api/v1/validation/file
```

#### Get Validation Report
```http
GET /api/v1/validation/report?limit=100&validation_type_filter=code
```

#### Batch Code Validation
```http
POST /api/v1/validation/batch/code
```

#### Get Placeholder Patterns
```http
GET /api/v1/validation/patterns/placeholders?language=python
```

#### Add Custom Validation Rules
```http
POST /api/v1/validation/rules/custom
```

#### Validation Service Health
```http
GET /api/v1/validation/health/service
```

### 4. AI Integration (`/ai`)

#### Generate Code with AI
```http
POST /api/v1/ai/code/generate
```

**Request Body:**
```json
{
  "prompt": "Create a FastAPI endpoint for user authentication",
  "language": "python",
  "context": {
    "framework": "fastapi",
    "database": "postgresql"
  },
  "validate_response": true,
  "use_cache": true
}
```

**Response:**
```json
{
  "code": "from fastapi import APIRouter, Depends...",
  "language": "python",
  "metadata": {
    "model": "claude-3",
    "tokens_used": 150
  },
  "validation": {
    "is_valid": true,
    "score": 0.95
  },
  "cached": false,
  "generated_at": "2025-01-01T00:00:00Z"
}
```

#### Orchestrate Task with Claude Flow
```http
POST /api/v1/ai/orchestrate
```

#### Validate AI Response
```http
POST /api/v1/ai/validate
```

#### Get AI Performance Metrics
```http
GET /api/v1/ai/performance
```

#### Get AI Request History
```http
GET /api/v1/ai/history?page=1&page_size=20&operation_filter=code_generation
```

#### Review Code with AI
```http
POST /api/v1/ai/code/review
```

#### Refactor Code with AI
```http
POST /api/v1/ai/code/refactor
```

#### Clear AI Cache
```http
DELETE /api/v1/ai/cache
```

#### Get AI Providers Status
```http
GET /api/v1/ai/providers/status
```

#### Test AI Connections
```http
POST /api/v1/ai/test/connection
```

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Validation failed",
  "status_code": 400,
  "details": {
    "field": "code",
    "message": "Code cannot be empty"
  },
  "timestamp": "2025-01-01T00:00:00Z",
  "request_id": "uuid-here"
}
```

## Common HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `408 Request Timeout` - Task execution timeout
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

## Response Headers

All responses include standard headers:

- `X-RateLimit-Limit` - Rate limit for the endpoint
- `X-RateLimit-Remaining` - Remaining requests in current window
- `X-RateLimit-Reset` - Unix timestamp when rate limit resets
- `X-Request-ID` - Unique request identifier for tracking

## Pagination

List endpoints support pagination:

```http
GET /api/v1/projects?page=1&page_size=10
```

Response includes pagination metadata:

```json
{
  "data": [...],
  "total": 100,
  "page": 1,
  "page_size": 10,
  "total_pages": 10
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

```http
GET /api/v1/tasks?task_type_filter=code_generation&status_filter=completed&sort=created_at:desc
```

## Batch Operations

Batch endpoints allow processing multiple items:

```http
POST /api/v1/validation/batch/code
POST /api/v1/tasks/batch/create
```

Batch responses include individual results:

```json
{
  "results": [
    {"success": true, "data": {...}},
    {"success": false, "error": "..."}
  ],
  "summary": {
    "total": 5,
    "successful": 4,
    "failed": 1
  }
}
```

## Health Checks

Each service provides health check endpoints:

```http
GET /api/v1/projects/health/service
GET /api/v1/tasks/health/service
GET /api/v1/validation/health/service
GET /api/v1/ai/providers/status
```

## API Versioning

The API uses URL-based versioning:

- Current version: `v1`
- Base path: `/api/v1`
- Future versions will be available at `/api/v2`, etc.

## OpenAPI Specification

Interactive API documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Support and Contact

For API support and questions:
- Documentation: https://docs.claude-tiu.example.com
- Issues: https://github.com/your-org/claude-tiu/issues
- Email: api-support@claude-tiu.example.com