#!/usr/bin/env python3
"""
Claude Integration Layer - Advanced Claude API Backend Services

Provides comprehensive integration with Claude Code and Claude Flow:
- Multi-model Claude API coordination
- Context-aware prompt engineering
- Advanced conversation management
- Streaming response handling
- Token optimization and billing tracking
- Rate limiting and retry logic
- Performance analytics and optimization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import hashlib

# Third-party imports
import aiohttp
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Internal imports
from .core_services import ServiceOrchestrator, get_service_orchestrator
from ..core.config_manager import ConfigManager
from ..claude_tui.integrations.claude_flow_client import ClaudeFlowClient
from ..claude_tui.integrations.claude_code_client import ClaudeCodeClient

logger = logging.getLogger(__name__)


class ClaudeModel(str, Enum):
    """Supported Claude models."""
    SONNET_4 = "claude-sonnet-4-20250514"
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    HAIKU_3_5 = "claude-3-5-haiku-20241022"
    OPUS_3 = "claude-3-opus-20240229"


class ConversationRole(str, Enum):
    """Conversation roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseMode(str, Enum):
    """Response generation modes."""
    COMPLETE = "complete"  # Wait for full response
    STREAMING = "streaming"  # Stream response as generated
    BATCH = "batch"  # Process multiple requests in batch


@dataclass
class ClaudeMessage:
    """Claude conversation message."""
    role: ConversationRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationContext:
    """Conversation context and state."""
    conversation_id: str
    messages: List[ClaudeMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model: ClaudeModel = ClaudeModel.SONNET_4
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Context tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    total_tokens_used: int = 0
    
    # Project context
    project_id: Optional[str] = None
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Performance tracking
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0


class ClaudeRequest(BaseModel):
    """Claude API request configuration."""
    prompt: str
    model: ClaudeModel = ClaudeModel.SONNET_4
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    stop_sequences: List[str] = []
    
    # Context
    conversation_id: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    
    # Options
    response_mode: ResponseMode = ResponseMode.COMPLETE
    priority: int = 5  # 1-10, higher is more priority
    timeout: int = 300  # 5 minutes
    retry_count: int = 3
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if not 1 <= v <= 8000:
            raise ValueError('Max tokens must be between 1 and 8000')
        return v


@dataclass
class ClaudeResponse:
    """Claude API response."""
    content: str
    model: ClaudeModel
    conversation_id: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    confidence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Error information
    error: Optional[str] = None
    retry_count: int = 0
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None and bool(self.content)


class TokenManager:
    """
    Manages token usage, billing, and optimization.
    """
    
    def __init__(self):
        self.usage_history: List[Dict[str, Any]] = []
        self.daily_usage: Dict[str, int] = {}
        self.model_costs = {
            ClaudeModel.SONNET_4: {"input": 0.003, "output": 0.015},  # per 1K tokens
            ClaudeModel.SONNET_3_5: {"input": 0.003, "output": 0.015},
            ClaudeModel.HAIKU_3_5: {"input": 0.00025, "output": 0.00125},
            ClaudeModel.OPUS_3: {"input": 0.015, "output": 0.075}
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return max(1, len(text) // 4)
    
    def calculate_cost(self, model: ClaudeModel, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        costs = self.model_costs.get(model, self.model_costs[ClaudeModel.SONNET_4])
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def record_usage(self, model: ClaudeModel, input_tokens: int, output_tokens: int, 
                    conversation_id: str, user_id: Optional[str] = None) -> None:
        """Record token usage."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        today = datetime.now().strftime('%Y-%m-%d')
        
        usage_record = {
            'timestamp': datetime.now().isoformat(),
            'model': model.value,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'conversation_id': conversation_id,
            'user_id': user_id
        }
        
        self.usage_history.append(usage_record)
        
        # Update daily usage
        if today not in self.daily_usage:
            self.daily_usage[today] = 0
        self.daily_usage[today] += input_tokens + output_tokens
    
    def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_usage = [
            record for record in self.usage_history
            if datetime.fromisoformat(record['timestamp']) > cutoff
        ]
        
        if not recent_usage:
            return {'total_tokens': 0, 'total_cost': 0.0, 'requests': 0}
        
        total_tokens = sum(record['total_tokens'] for record in recent_usage)
        total_cost = sum(record['cost'] for record in recent_usage)
        requests = len(recent_usage)
        
        model_usage = {}
        for record in recent_usage:
            model = record['model']
            if model not in model_usage:
                model_usage[model] = {'tokens': 0, 'cost': 0.0, 'requests': 0}
            model_usage[model]['tokens'] += record['total_tokens']
            model_usage[model]['cost'] += record['cost']
            model_usage[model]['requests'] += 1
        
        return {
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'requests': requests,
            'average_tokens_per_request': total_tokens / requests if requests > 0 else 0,
            'model_breakdown': model_usage,
            'daily_usage': dict(list(self.daily_usage.items())[-days:])
        }


class ContextOptimizer:
    """
    Optimizes conversation context to reduce token usage.
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.token_manager = TokenManager()
    
    def optimize_context(self, context: ConversationContext) -> ConversationContext:
        """Optimize conversation context to fit within token limits."""
        if not context.messages:
            return context
        
        # Estimate current token usage
        total_tokens = 0
        if context.system_prompt:
            total_tokens += self.token_manager.estimate_tokens(context.system_prompt)
        
        for message in context.messages:
            total_tokens += self.token_manager.estimate_tokens(message.content)
        
        # If within limits, return as-is
        if total_tokens <= self.max_context_tokens:
            return context
        
        # Optimize by removing older messages
        optimized_context = ConversationContext(
            conversation_id=context.conversation_id,
            system_prompt=context.system_prompt,
            model=context.model,
            temperature=context.temperature,
            max_tokens=context.max_tokens
        )
        
        # Keep system prompt
        remaining_tokens = self.max_context_tokens
        if context.system_prompt:
            remaining_tokens -= self.token_manager.estimate_tokens(context.system_prompt)
        
        # Add messages from newest to oldest until we hit the limit
        for message in reversed(context.messages):
            message_tokens = self.token_manager.estimate_tokens(message.content)
            if remaining_tokens - message_tokens > 0:
                optimized_context.messages.insert(0, message)
                remaining_tokens -= message_tokens
            else:
                break
        
        logger.info(f"Context optimized: {len(context.messages)} -> {len(optimized_context.messages)} messages")
        
        return optimized_context
    
    def summarize_old_messages(self, messages: List[ClaudeMessage]) -> str:
        """Summarize older messages to preserve context."""
        if not messages:
            return ""
        
        # Simple summarization - in practice, this could use Claude to summarize
        summary_parts = []
        for message in messages:
            if len(message.content) > 100:
                summary = message.content[:97] + "..."
            else:
                summary = message.content
            summary_parts.append(f"{message.role.value}: {summary}")
        
        return "Previous conversation summary: " + " | ".join(summary_parts)


class ClaudeIntegrationLayer:
    """
    Advanced Claude API integration layer.
    
    Features:
    - Multi-model support with intelligent routing
    - Conversation context management
    - Token optimization and cost tracking
    - Streaming response handling
    - Rate limiting and retry logic
    - Performance analytics
    - Integration with Claude Code and Claude Flow
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize Claude integration layer."""
        self.config_manager = config_manager
        
        # Service orchestrator
        self.orchestrator: Optional[ServiceOrchestrator] = None
        
        # Claude clients
        self.claude_code_client: Optional[ClaudeCodeClient] = None
        self.claude_flow_client: Optional[ClaudeFlowClient] = None
        
        # HTTP session for direct API calls
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Context management
        self.conversations: Dict[str, ConversationContext] = {}
        self.context_optimizer = ContextOptimizer()
        self.token_manager = TokenManager()
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            'requests_count': 0,
            'success_count': 0,
            'error_count': 0,
            'average_response_time': 0.0,
            'total_tokens_used': 0
        }
        
        # Configuration
        self.api_key: Optional[str] = None
        self.base_url = "https://api.anthropic.com"
        self.default_model = ClaudeModel.SONNET_4
        
        logger.info("Claude Integration Layer initialized")
    
    async def initialize(self) -> None:
        """Initialize the Claude integration layer."""
        logger.info("Initializing Claude Integration Layer...")
        
        try:
            # Get service orchestrator
            self.orchestrator = get_service_orchestrator()
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize HTTP session
            await self._initialize_http_session()
            
            # Initialize Claude clients
            await self._initialize_claude_clients()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_old_conversations())
            asyncio.create_task(self._performance_monitoring_loop())
            
            logger.info("Claude Integration Layer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Integration Layer: {e}")
            raise
    
    async def _load_configuration(self) -> None:
        """Load configuration settings."""
        claude_config = await self.config_manager.get_setting('claude', {})
        
        self.api_key = claude_config.get('api_key')
        self.base_url = claude_config.get('base_url', self.base_url)
        
        model_name = claude_config.get('default_model', 'sonnet-4')
        model_mapping = {
            'sonnet-4': ClaudeModel.SONNET_4,
            'sonnet-3.5': ClaudeModel.SONNET_3_5,
            'haiku-3.5': ClaudeModel.HAIKU_3_5,
            'opus-3': ClaudeModel.OPUS_3
        }
        self.default_model = model_mapping.get(model_name, ClaudeModel.SONNET_4)
        
        if not self.api_key:
            logger.warning("Claude API key not configured")
    
    async def _initialize_http_session(self) -> None:
        """Initialize HTTP session for API calls."""
        headers = {
            'User-Agent': 'Claude-TIU/0.1.0',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        
        self.http_session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=50,
                limit_per_host=10,
                ttl_dns_cache=300
            )
        )
    
    async def _initialize_claude_clients(self) -> None:
        """Initialize Claude Code and Claude Flow clients."""
        if self.orchestrator:
            # Get Claude Flow client from orchestrator
            self.claude_flow_client = self.orchestrator.get_claude_flow_service()
            
            # Initialize Claude Code client
            try:
                self.claude_code_client = ClaudeCodeClient(self.config_manager)
                await self.claude_code_client.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize Claude Code client: {e}")
    
    async def generate_response(
        self,
        request: ClaudeRequest,
        context: Optional[ConversationContext] = None
    ) -> ClaudeResponse:
        """
        Generate response using Claude API.
        
        Args:
            request: Claude API request configuration
            context: Optional conversation context
            
        Returns:
            Claude response
        """
        start_time = datetime.now()
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire(request.priority)
            
            # Get or create conversation context
            if context is None:
                context = self.conversations.get(
                    conversation_id,
                    ConversationContext(conversation_id=conversation_id, model=request.model)
                )
            
            # Optimize context to fit token limits
            context = self.context_optimizer.optimize_context(context)
            
            # Add user message to context
            user_message = ClaudeMessage(
                role=ConversationRole.USER,
                content=request.prompt,
                metadata={
                    'project_context': request.project_context,
                    'user_context': request.user_context
                }
            )
            context.messages.append(user_message)
            
            # Generate response based on mode
            if request.response_mode == ResponseMode.STREAMING:
                return await self._generate_streaming_response(request, context)
            else:
                return await self._generate_complete_response(request, context)
                
        except Exception as e:
            logger.error(f"Failed to generate Claude response: {e}")
            
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(success=False, response_time=response_time)
            
            return ClaudeResponse(
                content="",
                model=request.model,
                conversation_id=conversation_id,
                error=str(e),
                response_time=response_time
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    async def _generate_complete_response(
        self,
        request: ClaudeRequest,
        context: ConversationContext
    ) -> ClaudeResponse:
        """Generate complete response using Claude API."""
        start_time = datetime.now()
        
        # Prepare API request
        api_request = self._prepare_api_request(request, context)
        
        # Make API call
        async with self.http_session.post(
            f"{self.base_url}/v1/messages",
            json=api_request,
            timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            response.raise_for_status()
            response_data = await response.json()
        
        # Parse response
        content = response_data.get('content', [{}])[0].get('text', '')
        usage = response_data.get('usage', {})
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Create Claude response
        claude_response = ClaudeResponse(
            content=content,
            model=request.model,
            conversation_id=context.conversation_id,
            response_time=response_time,
            token_usage={
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0)
            }
        )
        
        # Add assistant message to context
        assistant_message = ClaudeMessage(
            role=ConversationRole.ASSISTANT,
            content=content,
            token_count=usage.get('output_tokens', 0)
        )
        context.messages.append(assistant_message)
        
        # Update context
        context.last_interaction = datetime.now()
        context.response_times.append(response_time)
        context.total_tokens_used += usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        
        # Store updated context
        self.conversations[context.conversation_id] = context
        
        # Record token usage
        self.token_manager.record_usage(
            request.model,
            usage.get('input_tokens', 0),
            usage.get('output_tokens', 0),
            context.conversation_id
        )
        
        # Update performance metrics
        self._update_performance_metrics(
            success=True,
            response_time=response_time,
            tokens_used=usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        )
        
        return claude_response
    
    async def _generate_streaming_response(
        self,
        request: ClaudeRequest,
        context: ConversationContext
    ) -> ClaudeResponse:
        """Generate streaming response using Claude API."""
        # Streaming implementation would go here
        # For now, fall back to complete response
        logger.info("Streaming mode requested, falling back to complete response")
        return await self._generate_complete_response(request, context)
    
    def _prepare_api_request(self, request: ClaudeRequest, context: ConversationContext) -> Dict[str, Any]:
        """Prepare API request payload."""
        messages = []
        
        # Add conversation history
        for message in context.messages:
            messages.append({
                'role': message.role.value,
                'content': message.content
            })
        
        api_request = {
            'model': request.model.value,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'messages': messages
        }
        
        if request.system_prompt or context.system_prompt:
            api_request['system'] = request.system_prompt or context.system_prompt
        
        if request.stop_sequences:
            api_request['stop_sequences'] = request.stop_sequences
        
        return api_request
    
    async def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[ClaudeModel] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Create a new conversation context."""
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            model=model or self.default_model,
            project_id=project_id,
            user_id=user_id
        )
        
        self.conversations[conversation_id] = context
        
        logger.info(f"Created conversation {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self.conversations.get(conversation_id)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation context."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        return False
    
    async def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary."""
        context = self.conversations.get(conversation_id)
        if not context:
            return None
        
        return {
            'conversation_id': conversation_id,
            'created_at': context.created_at.isoformat(),
            'last_interaction': context.last_interaction.isoformat(),
            'message_count': len(context.messages),
            'total_tokens_used': context.total_tokens_used,
            'model': context.model.value,
            'average_response_time': sum(context.response_times) / len(context.response_times) if context.response_times else 0,
            'project_id': context.project_id,
            'user_id': context.user_id
        }
    
    async def integrate_with_claude_flow(
        self,
        conversation_id: str,
        workflow_name: str,
        parameters: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Integrate conversation with Claude Flow workflow."""
        if not self.claude_flow_client:
            logger.warning("Claude Flow client not available")
            return None
        
        try:
            from ..claude_tui.models.ai_models import WorkflowRequest
            
            context = self.conversations.get(conversation_id)
            if not context:
                logger.error(f"Conversation {conversation_id} not found")
                return None
            
            # Prepare workflow request with conversation context
            workflow_request = WorkflowRequest(
                workflow_name=workflow_name,
                parameters=parameters or {},
                variables={
                    'conversation_id': conversation_id,
                    'message_count': len(context.messages),
                    'model': context.model.value,
                    'project_id': context.project_id
                }
            )
            
            # Execute workflow
            result = await self.claude_flow_client.execute_workflow(workflow_request)
            
            logger.info(f"Claude Flow workflow '{workflow_name}' executed for conversation {conversation_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to integrate with Claude Flow: {e}")
            return None
    
    def _update_performance_metrics(self, success: bool, response_time: float, tokens_used: int = 0) -> None:
        """Update performance metrics."""
        self.performance_metrics['requests_count'] += 1
        
        if success:
            self.performance_metrics['success_count'] += 1
        else:
            self.performance_metrics['error_count'] += 1
        
        # Update average response time
        total_requests = self.performance_metrics['requests_count']
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        self.performance_metrics['total_tokens_used'] += tokens_used
    
    async def _cleanup_old_conversations(self) -> None:
        """Clean up old conversation contexts."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff = datetime.now() - timedelta(hours=24)  # Keep conversations for 24 hours
                to_remove = []
                
                for conv_id, context in self.conversations.items():
                    if context.last_interaction < cutoff:
                        to_remove.append(conv_id)
                
                for conv_id in to_remove:
                    del self.conversations[conv_id]
                    logger.info(f"Cleaned up old conversation {conv_id}")
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old conversations")
                    
            except Exception as e:
                logger.error(f"Error in conversation cleanup: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor performance and log metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Log current performance metrics
                metrics = self.get_performance_metrics()
                logger.info(f"Claude Integration Performance: {json.dumps(metrics, indent=2)}")
                
                # Store metrics in cache if available
                if self.orchestrator:
                    cache_service = self.orchestrator.get_cache_service()
                    if cache_service:
                        await cache_service.set(
                            f"claude_metrics:{datetime.now().strftime('%Y%m%d%H%M')}",
                            metrics,
                            ttl=3600
                        )
                        
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add additional metrics
        metrics.update({
            'active_conversations': len(self.conversations),
            'success_rate': (
                self.performance_metrics['success_count'] / 
                max(1, self.performance_metrics['requests_count'])
            ) * 100,
            'token_usage_stats': self.token_manager.get_usage_stats(),
            'timestamp': datetime.now().isoformat()
        })
        
        return metrics
    
    async def cleanup(self) -> None:
        """Clean up Claude integration resources."""
        logger.info("Cleaning up Claude Integration Layer...")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        # Clear conversations
        self.conversations.clear()
        
        logger.info("Claude Integration Layer cleanup completed")


class RateLimiter:
    """Rate limiter for Claude API requests."""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_update = datetime.now()
        self.lock = asyncio.Lock()
    
    async def acquire(self, priority: int = 5) -> None:
        """Acquire rate limit token."""
        async with self.lock:
            now = datetime.now()
            time_passed = (now - self.last_update).total_seconds()
            
            # Add tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)
            self.last_update = now
            
            # Wait if no tokens available
            while self.tokens < 1:
                await asyncio.sleep(0.1)
                now = datetime.now()
                time_passed = (now - self.last_update).total_seconds()
                tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
                self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)
                self.last_update = now
            
            # Consume token
            self.tokens -= 1


# Global integration layer instance
claude_integration: Optional[ClaudeIntegrationLayer] = None


def get_claude_integration() -> Optional[ClaudeIntegrationLayer]:
    """Get the global Claude integration instance."""
    return claude_integration


async def initialize_claude_integration(config_manager: ConfigManager) -> ClaudeIntegrationLayer:
    """Initialize the global Claude integration layer."""
    global claude_integration
    
    if claude_integration is None:
        claude_integration = ClaudeIntegrationLayer(config_manager)
        await claude_integration.initialize()
    
    return claude_integration


async def cleanup_claude_integration() -> None:
    """Clean up the global Claude integration layer."""
    global claude_integration
    
    if claude_integration is not None:
        await claude_integration.cleanup()
        claude_integration = None
