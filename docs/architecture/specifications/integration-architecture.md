# Integration Architecture Specification
## Intelligent Claude-TUI Component Integration Design

> **Integration Architect**: Component integration patterns design  
> **Integration with**: System Architecture v1.0.0, Data Architecture v1.0.0  
> **Date**: August 25, 2025  

---

## 🎯 Integration Architecture Overview

The integration architecture defines how Claude-TUI, Claude-Flow, Claude-Code, and MCP server components communicate, coordinate, and exchange data in real-time to create a unified intelligent system.

---

## 🔗 Component Integration Model

### Integration Topology
```
┌─────────────────────────────────────────────────────────────────┐
│                 INTEGRATION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │   CLAUDE-TUI    │◄──────► │  CLAUDE-FLOW    │               │
│  │ (TUI Interface) │   WS    │ (Orchestrator)  │               │
│  └─────────────────┘  /HTTP  └─────────────────┘               │
│           │                           │                        │
│           │ MCP                      │ REST                    │
│           ▼                           ▼                        │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │   MCP SERVER    │◄──────► │  CLAUDE-CODE    │               │
│  │ (Tool Gateway)  │   HTTP  │  (AI Engine)    │               │
│  └─────────────────┘         └─────────────────┘               │
│           │                           │                        │
│           └─────────── EVENT BUS ──────┘                       │
│                          │                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   MESSAGE BROKER                            │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │ │
│  │  │   Redis    │  │  WebSocket │  │    HTTP     │          │ │
│  │  │  PubSub    │  │   Server   │  │   Gateway   │          │ │
│  │  └────────────┘  └────────────┘  └────────────┘          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📡 Communication Protocols

### Protocol Matrix
```
Component Pair          | Protocol    | Port  | Purpose                | Latency
──────────────────────── | ─────────── | ───── | ──────────────────── | ────────
TUI ↔ Flow              | WebSocket   | 8080  | Real-time commands     | < 10ms
TUI ↔ MCP               | MCP/STDIO   | -     | Tool integration       | < 50ms
Flow ↔ Code             | HTTP/REST   | 8081  | AI processing          | < 100ms
Flow ↔ MCP              | HTTP        | 8082  | Tool orchestration     | < 50ms
Code ↔ MCP              | WebHooks    | 8083  | Event callbacks        | < 25ms
All ↔ EventBus          | Redis PubSub| 6379  | Event broadcasting     | < 5ms
```

### WebSocket Protocol (TUI ↔ Flow)
```json
{
  "protocol": "claude-tui-flow",
  "version": "1.0.0",
  "message_types": {
    "command": {
      "type": "command",
      "id": "uuid",
      "timestamp": "iso8601",
      "payload": {
        "action": "string",
        "parameters": "object",
        "session_id": "string"
      }
    },
    "response": {
      "type": "response", 
      "id": "uuid",
      "request_id": "uuid",
      "timestamp": "iso8601",
      "status": "success|error|pending",
      "payload": "object"
    },
    "event": {
      "type": "event",
      "event_name": "string", 
      "timestamp": "iso8601",
      "payload": "object"
    },
    "stream": {
      "type": "stream",
      "stream_id": "uuid",
      "chunk_id": "number",
      "payload": "object",
      "complete": "boolean"
    }
  }
}
```

### HTTP/REST Protocol (Flow ↔ Code)
```yaml
API_Specification:
  base_url: "http://claude-code:8081/api/v1"
  authentication: "bearer_token"
  content_type: "application/json"
  
  endpoints:
    process_request:
      method: POST
      path: "/process"
      payload:
        context: "string"
        instruction: "string" 
        session_id: "string"
        parameters: "object"
      response:
        result: "object"
        confidence: "number"
        processing_time: "number"
        
    get_capabilities:
      method: GET
      path: "/capabilities"
      response:
        available_models: "array"
        supported_tasks: "array"
        resource_limits: "object"
        
    stream_process:
      method: POST
      path: "/stream"
      response_type: "text/event-stream"
      events:
        - progress
        - partial_result
        - complete
        - error
```

### MCP Protocol Integration
```python
class MCPIntegration:
    def __init__(self):
        self.server_config = {
            "name": "claude-tui-mcp",
            "version": "1.0.0",
            "protocol_version": "2024-11-05"
        }
    
    def register_tools(self):
        return {
            "memory_store": {
                "description": "Store/retrieve collective memory",
                "parameters": {
                    "action": {"type": "string", "enum": ["store", "retrieve"]},
                    "key": {"type": "string"},
                    "value": {"type": "object", "optional": True}
                }
            },
            "agent_spawn": {
                "description": "Spawn new agent in swarm",
                "parameters": {
                    "agent_type": {"type": "string"},
                    "capabilities": {"type": "array"},
                    "priority": {"type": "number"}
                }
            },
            "task_orchestrate": {
                "description": "Orchestrate complex tasks",
                "parameters": {
                    "task_definition": {"type": "object"},
                    "dependencies": {"type": "array"},
                    "execution_plan": {"type": "object"}
                }
            }
        }
```

---

## 🔄 Event-Driven Architecture

### Event Bus Design
```python
class EventBus:
    def __init__(self):
        self.redis = redis.Redis()
        self.channels = {
            'system': 'claude-tui:system',
            'agent': 'claude-tui:agent',
            'task': 'claude-tui:task',
            'memory': 'claude-tui:memory',
            'ui': 'claude-tui:ui'
        }
    
    async def publish(self, channel: str, event: dict):
        """Publish event to channel"""
        event_data = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'source': self.component_id,
            'type': event.get('type'),
            'payload': event.get('payload', {})
        }
        
        await self.redis.publish(
            self.channels[channel], 
            json.dumps(event_data)
        )
    
    async def subscribe(self, channel: str, callback: callable):
        """Subscribe to channel events"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channels[channel])
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                event = json.loads(message['data'])
                await callback(event)
```

### Event Types & Handlers
```yaml
EventTypes:
  system_events:
    - system_startup
    - system_shutdown
    - component_ready
    - health_check
    
  agent_events:
    - agent_spawned
    - agent_terminated  
    - agent_status_changed
    - agent_error
    
  task_events:
    - task_created
    - task_started
    - task_completed
    - task_failed
    - task_progress
    
  memory_events:
    - memory_updated
    - pattern_discovered
    - cache_invalidated
    - backup_completed
    
  ui_events:
    - user_input
    - display_updated
    - view_changed
    - error_displayed

EventHandlers:
  claude-tui:
    subscribes: [system, task, ui]
    publishes: [ui, task]
    
  claude-flow:
    subscribes: [system, agent, task]
    publishes: [agent, task, system]
    
  claude-code:  
    subscribes: [task]
    publishes: [task, memory]
    
  mcp-server:
    subscribes: [system, memory]
    publishes: [system, memory]
```

---

## 🔌 Plugin Architecture

### Plugin Framework
```python
class PluginFramework:
    def __init__(self):
        self.plugins = {}
        self.plugin_registry = PluginRegistry()
        self.event_bus = EventBus()
    
    def register_plugin(self, plugin: Plugin):
        """Register a new plugin"""
        plugin_info = {
            'id': plugin.get_id(),
            'version': plugin.get_version(),
            'capabilities': plugin.get_capabilities(),
            'dependencies': plugin.get_dependencies(),
            'endpoints': plugin.get_endpoints()
        }
        
        self.plugin_registry.register(plugin_info)
        self.plugins[plugin.get_id()] = plugin
        
        # Setup plugin event subscriptions
        for event_type in plugin.get_subscriptions():
            self.event_bus.subscribe(
                event_type, 
                plugin.handle_event
            )
    
    async def invoke_plugin(self, plugin_id: str, method: str, **kwargs):
        """Invoke plugin method"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise PluginNotFoundError(f"Plugin {plugin_id} not found")
        
        return await plugin.invoke(method, **kwargs)
```

### Plugin Interface Standard
```python
class Plugin(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """Return unique plugin identifier"""
        pass
    
    @abstractmethod  
    def get_version(self) -> str:
        """Return plugin version"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of plugin capabilities"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Return list of required dependencies"""
        pass
    
    @abstractmethod  
    def get_endpoints(self) -> Dict[str, callable]:
        """Return dict of endpoint name -> handler"""
        pass
    
    @abstractmethod
    def get_subscriptions(self) -> List[str]:
        """Return list of event types to subscribe to"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: dict):
        """Handle incoming events"""
        pass
    
    @abstractmethod
    async def invoke(self, method: str, **kwargs):
        """Invoke plugin method"""
        pass
```

---

## 🌐 API Gateway Configuration

### Gateway Design
```python
class APIGateway:
    def __init__(self):
        self.routes = {}
        self.middleware = []
        self.rate_limiter = RateLimiter()
        self.auth_handler = AuthenticationHandler()
    
    def configure_routes(self):
        self.routes = {
            # Claude-TUI routes
            '/tui/command': {
                'target': 'http://claude-tui:8000',
                'methods': ['POST'],
                'auth_required': True,
                'rate_limit': '1000/minute'
            },
            
            # Claude-Flow routes  
            '/flow/orchestrate': {
                'target': 'http://claude-flow:8080',
                'methods': ['POST'],
                'auth_required': True,
                'rate_limit': '500/minute'
            },
            
            # Claude-Code routes
            '/code/process': {
                'target': 'http://claude-code:8081', 
                'methods': ['POST'],
                'auth_required': True,
                'rate_limit': '100/minute'
            },
            
            # MCP routes
            '/mcp/tools': {
                'target': 'http://mcp-server:8082',
                'methods': ['GET', 'POST'],
                'auth_required': True,
                'rate_limit': '2000/minute'
            }
        }
    
    async def handle_request(self, request):
        # Apply middleware
        for middleware in self.middleware:
            request = await middleware.process(request)
        
        # Authenticate
        if self.routes[request.path]['auth_required']:
            await self.auth_handler.authenticate(request)
        
        # Rate limiting
        await self.rate_limiter.check_limits(request)
        
        # Route to target service
        return await self.proxy_request(request)
```

### Load Balancing Strategy
```yaml
LoadBalancing:
  algorithm: "weighted_round_robin"
  health_checks:
    enabled: true
    interval: "30s"
    timeout: "5s"
    failure_threshold: 3
  
  service_weights:
    claude-tui: 1.0
    claude-flow: 2.0    # Higher weight - orchestration heavy
    claude-code: 0.5    # Lower weight - AI processing intensive
    mcp-server: 1.5
  
  failover:
    enabled: true
    backup_instances: 2
    recovery_time: "60s"
```

---

## 🔐 Security Integration

### Authentication Flow
```
User Request → API Gateway → Auth Handler → JWT Validation → Service Routing
     │               │            │              │               │
     ▼               ▼            ▼              ▼               ▼
  Credentials    Rate Limiting  Token Check  Permission    Target Service
     │               │            │          Validation         │
     └─────────── Security Log ───┴──────────────┴──────────────┘
```

### Authorization Matrix  
```yaml
Permissions:
  user:
    claude-tui: [read, write, execute]
    claude-flow: [read, execute] 
    claude-code: [execute]
    mcp-server: [read, execute]
  
  admin:
    claude-tui: [read, write, execute, configure]
    claude-flow: [read, write, execute, configure]
    claude-code: [read, write, execute, configure]
    mcp-server: [read, write, execute, configure]
  
  service:
    claude-tui: [system]
    claude-flow: [system, orchestrate] 
    claude-code: [system, process]
    mcp-server: [system, tools]
```

---

## 📊 Integration Monitoring

### Health Check System
```python
class HealthMonitor:
    def __init__(self):
        self.services = ['claude-tui', 'claude-flow', 'claude-code', 'mcp-server']
        self.checks = {}
    
    async def run_health_checks(self):
        """Run health checks for all services"""
        results = {}
        
        for service in self.services:
            try:
                result = await self.check_service_health(service)
                results[service] = {
                    'status': 'healthy' if result.ok else 'unhealthy',
                    'response_time': result.response_time,
                    'last_check': datetime.utcnow().isoformat()
                }
            except Exception as e:
                results[service] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.utcnow().isoformat()
                }
        
        return results
    
    async def check_service_health(self, service: str):
        """Check individual service health"""
        health_endpoints = {
            'claude-tui': 'http://claude-tui:8000/health',
            'claude-flow': 'http://claude-flow:8080/health', 
            'claude-code': 'http://claude-code:8081/health',
            'mcp-server': 'http://mcp-server:8082/health'
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                health_endpoints[service], 
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                end_time = time.time()
                
                return HealthResult(
                    ok=response.status == 200,
                    status_code=response.status,
                    response_time=end_time - start_time,
                    payload=await response.json() if response.status == 200 else None
                )
```

### Performance Metrics Collection
```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.collectors = [
            ResponseTimeCollector(),
            ThroughputCollector(), 
            ErrorRateCollector(),
            ResourceUsageCollector()
        ]
    
    async def collect_metrics(self):
        """Collect metrics from all services"""
        for collector in self.collectors:
            metrics = await collector.collect()
            self.metrics.update(metrics)
        
        return self.metrics
    
    def get_integration_metrics(self):
        """Get integration-specific metrics"""
        return {
            'message_throughput': self.metrics.get('messages_per_second', 0),
            'average_latency': self.metrics.get('average_response_time', 0),
            'error_rate': self.metrics.get('error_percentage', 0),
            'active_connections': self.metrics.get('websocket_connections', 0),
            'plugin_count': self.metrics.get('active_plugins', 0)
        }
```

---

## 🔄 Data Exchange Formats

### Standard Message Schema
```json
{
  "$schema": "https://claude-tui.com/schemas/message/v1.0.0.json",
  "type": "object",
  "properties": {
    "header": {
      "type": "object",
      "properties": {
        "id": {"type": "string", "format": "uuid"},
        "timestamp": {"type": "string", "format": "date-time"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "source": {"type": "string"},
        "target": {"type": "string"},
        "correlation_id": {"type": "string", "format": "uuid"}
      },
      "required": ["id", "timestamp", "version", "source"]
    },
    "payload": {
      "type": "object",
      "properties": {
        "type": {"type": "string"},
        "data": {"type": "object"},
        "metadata": {"type": "object"}
      },
      "required": ["type", "data"]
    }
  },
  "required": ["header", "payload"]
}
```

### Error Handling Protocol
```python
class ErrorHandler:
    def __init__(self):
        self.error_types = {
            'ValidationError': 400,
            'AuthenticationError': 401,
            'AuthorizationError': 403,
            'NotFoundError': 404,
            'RateLimitError': 429,
            'ServiceUnavailableError': 503,
            'IntegrationError': 502
        }
    
    def format_error_response(self, error: Exception, request_id: str):
        """Format standardized error response"""
        error_type = type(error).__name__
        status_code = self.error_types.get(error_type, 500)
        
        return {
            'header': {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'source': 'integration-layer',
                'correlation_id': request_id
            },
            'payload': {
                'type': 'error',
                'data': {
                    'error_type': error_type,
                    'error_message': str(error),
                    'status_code': status_code,
                    'retry_after': self.get_retry_delay(error_type)
                },
                'metadata': {
                    'recoverable': self.is_recoverable(error_type),
                    'suggested_action': self.get_suggested_action(error_type)
                }
            }
        }
```

---

## 🎯 Implementation Roadmap

### Phase 1: Core Integration ✅
- ✅ WebSocket communication (TUI ↔ Flow)
- ✅ HTTP REST APIs (Flow ↔ Code)  
- ✅ MCP protocol implementation
- ✅ Basic event bus

### Phase 2: Advanced Features 🔄
- 🔄 Plugin framework
- 🔄 API gateway
- 🔄 Load balancing
- 🔄 Security integration

### Phase 3: Optimization ⏳
- ⏳ Performance monitoring
- ⏳ Auto-scaling
- ⏳ Circuit breakers
- ⏳ Advanced error handling

### Phase 4: Enterprise Features ⏳
- ⏳ Multi-tenant support
- ⏳ Service mesh integration
- ⏳ Advanced security
- ⏳ Compliance features

---

## 📈 Success Metrics

### Integration Performance
- **Message Throughput**: > 10,000 msg/sec
- **Average Latency**: < 50ms
- **Error Rate**: < 0.1%
- **Availability**: > 99.95%

### Protocol Efficiency  
- **WebSocket Connections**: > 1,000 concurrent
- **HTTP Request Rate**: > 5,000 req/sec
- **Plugin Load Time**: < 100ms
- **Service Discovery**: < 10ms

---

*Integration Architecture designed by: Integration Architect Team*  
*Coordinated with: System Architecture, Data Architecture*  
*Next: UI/UX Architecture and final validation*