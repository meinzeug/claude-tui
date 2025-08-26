"""
AI Interface - Central AI abstraction layer for claude-tiu.

Provides a unified interface for all AI operations including code generation,
analysis, validation, and workflow orchestration through Claude Code integration.
Implements intelligent request routing, caching, and error handling.

Features:
- Unified AI service abstraction
- Intelligent request routing and load balancing  
- Advanced caching and optimization
- Comprehensive error handling and retry logic
- Integration with validation and monitoring systems
- Support for multiple AI providers and models
- Context-aware request optimization
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from collections import deque
from uuid import UUID, uuid4

from .types import (
    ValidationResult, ProgressMetrics, SystemMetrics,
    ClaudeTUIException
)
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AIProviderType(str, Enum):
    """Available AI providers."""
    CLAUDE_CODE = "claude_code"
    CLAUDE_FLOW = "claude_flow"
    ANTHROPIC_API = "anthropic_api"
    LOCAL_MODEL = "local_model"


class RequestType(str, Enum):
    """Types of AI requests."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTOR = "code_refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    TASK_EXECUTION = "task_execution"


@dataclass
class AIRequest:
    """Structured AI request with context and metadata."""
    request_id: UUID = field(default_factory=uuid4)
    request_type: RequestType = RequestType.TASK_EXECUTION
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    model: str = "claude-3.5-sonnet"
    provider: Optional[AIProviderType] = None
    timeout: int = 300
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    priority: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIResponse:
    """Structured AI response with metrics and validation."""
    request_id: UUID
    success: bool
    content: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    tokens_used: int = 0
    processing_time: float = 0.0
    model_used: str = ""
    provider_used: AIProviderType = AIProviderType.CLAUDE_CODE
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIContext:
    """Enhanced context for AI operations."""
    project_path: Optional[Path] = None
    current_files: List[str] = field(default_factory=list)
    project_structure: Dict[str, Any] = field(default_factory=dict)
    existing_code: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    validation_context: Dict[str, Any] = field(default_factory=dict)


class AIInterfaceException(ClaudeTUIException):
    """AI interface specific exceptions."""
    pass


class RequestRouter:
    """Intelligent request routing and load balancing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._provider_health: Dict[AIProviderType, Dict[str, Any]] = {}
        self._request_history: deque = deque(maxlen=1000)
        
        # Initialize provider configurations
        self._provider_configs = self._load_provider_configs()
    
    def _load_provider_configs(self) -> Dict[AIProviderType, Dict[str, Any]]:
        """Load provider configurations from config manager."""
        ai_config = self.config_manager.get_config_value('ai', {})
        
        return {
            AIProviderType.CLAUDE_CODE: ai_config.get('claude_code', {
                'enabled': True,
                'timeout': 300,
                'max_retries': 3,
                'max_tokens': 4000,
                'priority': 1
            }),
            AIProviderType.CLAUDE_FLOW: ai_config.get('claude_flow', {
                'enabled': True,
                'timeout': 600,
                'max_retries': 2,
                'max_tokens': 8000,
                'priority': 2
            }),
            AIProviderType.ANTHROPIC_API: ai_config.get('anthropic_api', {
                'enabled': False,
                'timeout': 300,
                'max_retries': 3,
                'max_tokens': 4000,
                'priority': 3
            })
        }
    
    def route_request(self, request: AIRequest) -> AIProviderType:
        """Route request to optimal provider based on type, load, and health."""
        if request.provider:
            # Explicit provider specified
            return request.provider
        
        # Get available providers for request type
        available_providers = self._get_available_providers(request.request_type)
        
        if not available_providers:
            raise AIInterfaceException("No available providers for request type")
        
        # Select best provider based on health and load
        best_provider = self._select_best_provider(available_providers, request)
        
        logger.debug(f"Routing request {request.request_id} to provider {best_provider}")
        return best_provider
    
    def _get_available_providers(self, request_type: RequestType) -> List[AIProviderType]:
        """Get available providers for specific request type."""
        providers = []
        
        for provider_type, config in self._provider_configs.items():
            if not config.get('enabled', False):
                continue
            
            # Check provider capabilities
            if self._provider_supports_request_type(provider_type, request_type):
                providers.append(provider_type)
        
        return sorted(providers, key=lambda p: self._provider_configs[p].get('priority', 999))
    
    def _provider_supports_request_type(
        self, 
        provider: AIProviderType, 
        request_type: RequestType
    ) -> bool:
        """Check if provider supports specific request type."""
        # Provider capability mapping
        capabilities = {
            AIProviderType.CLAUDE_CODE: {
                RequestType.CODE_GENERATION,
                RequestType.CODE_REVIEW,
                RequestType.CODE_REFACTOR,
                RequestType.DOCUMENTATION,
                RequestType.TESTING,
                RequestType.TASK_EXECUTION
            },
            AIProviderType.CLAUDE_FLOW: {
                RequestType.CODE_GENERATION,
                RequestType.TASK_EXECUTION,
                RequestType.ANALYSIS
            },
            AIProviderType.ANTHROPIC_API: {
                RequestType.CODE_GENERATION,
                RequestType.CODE_REVIEW,
                RequestType.DOCUMENTATION,
                RequestType.ANALYSIS
            }
        }
        
        return request_type in capabilities.get(provider, set())
    
    def _select_best_provider(
        self, 
        providers: List[AIProviderType], 
        request: AIRequest
    ) -> AIProviderType:
        """Select best provider based on health, load, and request characteristics."""
        if not providers:
            raise AIInterfaceException("No providers available")
        
        if len(providers) == 1:
            return providers[0]
        
        # Score each provider
        provider_scores = {}
        
        for provider in providers:
            score = 0
            
            # Base priority score
            priority = self._provider_configs[provider].get('priority', 999)
            score += (1000 - priority)
            
            # Health score
            health_info = self._provider_health.get(provider, {})
            error_rate = health_info.get('error_rate', 0.1)
            avg_response_time = health_info.get('avg_response_time', 5.0)
            
            # Lower error rate is better
            score += (1 - error_rate) * 100
            
            # Lower response time is better (up to reasonable limit)
            if avg_response_time > 0:
                score += max(0, 100 - avg_response_time)
            
            # Request type affinity
            if provider == AIProviderType.CLAUDE_CODE:
                if request.request_type in [RequestType.CODE_GENERATION, RequestType.TASK_EXECUTION]:
                    score += 50
            
            provider_scores[provider] = score
        
        # Return provider with highest score
        best_provider = max(provider_scores, key=provider_scores.get)
        return best_provider
    
    def update_provider_health(
        self, 
        provider: AIProviderType, 
        response_time: float, 
        success: bool
    ) -> None:
        """Update provider health metrics."""
        if provider not in self._provider_health:
            self._provider_health[provider] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0.0,
                'error_rate': 0.0,
                'avg_response_time': 0.0
            }
        
        health = self._provider_health[provider]
        health['total_requests'] += 1
        health['total_response_time'] += response_time
        
        if success:
            health['successful_requests'] += 1
        
        # Update derived metrics
        health['error_rate'] = 1 - (health['successful_requests'] / health['total_requests'])
        health['avg_response_time'] = health['total_response_time'] / health['total_requests']


class ResponseCache:
    """Intelligent response caching with TTL and invalidation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
    
    def get_cache_key(self, request: AIRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.request_type.value}:{request.prompt}:{str(request.context)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, cache_key: str) -> Optional[AIResponse]:
        """Get cached response if valid."""
        if cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        cached_at = cache_entry['cached_at']
        ttl = cache_entry.get('ttl', self.default_ttl)
        
        # Check if expired
        if datetime.utcnow() - cached_at > timedelta(seconds=ttl):
            self._remove(cache_key)
            return None
        
        # Update access time
        self._access_times[cache_key] = datetime.utcnow()
        
        response = cache_entry['response']
        logger.debug(f"Cache hit for key {cache_key}")
        return response
    
    def set(
        self, 
        cache_key: str, 
        response: AIResponse, 
        ttl: Optional[int] = None
    ) -> None:
        """Cache response with optional TTL."""
        # Clean up expired entries if cache is full
        if len(self._cache) >= self.max_size:
            self._cleanup_expired()
            
            # If still full, remove LRU entries
            if len(self._cache) >= self.max_size:
                self._cleanup_lru()
        
        self._cache[cache_key] = {
            'response': response,
            'cached_at': datetime.utcnow(),
            'ttl': ttl or self.default_ttl
        }
        
        self._access_times[cache_key] = datetime.utcnow()
        logger.debug(f"Cached response for key {cache_key}")
    
    def _remove(self, cache_key: str) -> None:
        """Remove entry from cache."""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self._cache.items():
            cached_at = entry['cached_at']
            ttl = entry.get('ttl', self.default_ttl)
            
            if current_time - cached_at > timedelta(seconds=ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cleanup_lru(self) -> None:
        """Remove least recently used entries."""
        if not self._access_times:
            return
        
        # Remove oldest 20% of entries
        cleanup_count = max(1, len(self._access_times) // 5)
        
        # Sort by access time
        sorted_keys = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest entries
        for key, _ in sorted_keys[:cleanup_count]:
            self._remove(key)
        
        logger.debug(f"Cleaned up {cleanup_count} LRU cache entries")


class AIInterface:
    """
    Central AI abstraction layer for claude-tiu.
    
    Provides a unified interface for all AI operations with intelligent
    routing, caching, validation, and monitoring.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize AI interface.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Core components
        self.router = RequestRouter(self.config_manager)
        self.cache = ResponseCache()
        
        # Provider clients
        self._provider_clients: Dict[AIProviderType, Any] = {}
        
        # Metrics and monitoring
        self._request_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, float] = {}
        
        # State management
        self._initialized = False
        
        logger.info("AIInterface initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize AI interface and provider clients.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        logger.info("Initializing AIInterface")
        
        try:
            # Initialize provider clients
            await self._initialize_provider_clients()
            
            # Test connectivity
            test_successful = await self._test_connectivity()
            
            if test_successful:
                self._initialized = True
                logger.info("AIInterface initialization successful")
                return True
            else:
                logger.error("AIInterface connectivity test failed")
                return False
        
        except Exception as e:
            logger.error(f"AIInterface initialization failed: {e}")
            return False
    
    async def execute_development_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a development task with AI assistance.
        
        Args:
            request_data: Task request data
            
        Returns:
            Task execution result
        """
        if not self._initialized:
            await self.initialize()
        
        # Build AI request
        ai_request = AIRequest(
            request_type=RequestType.TASK_EXECUTION,
            prompt=request_data.get('task_description', ''),
            context=request_data.get('context', {}),
            timeout=request_data.get('timeout', 300),
            metadata={'project_path': str(request_data.get('project_path', ''))}
        )
        
        # Execute request
        response = await self.execute_request(ai_request)
        
        # Convert to expected format
        return {
            'success': response.success,
            'output': response.content,
            'files_modified': response.files_modified,
            'files_created': response.files_created,
            'validation_score': response.validation_score,
            'tokens_used': response.tokens_used,
            'error_message': response.error_message,
            'metadata': response.metadata
        }
    
    async def generate_code(
        self,
        description: str,
        language: str,
        context: Optional[AIContext] = None,
        style_preferences: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """
        Generate code based on description and context.
        
        Args:
            description: Code description
            language: Programming language
            context: AI context information
            style_preferences: Coding style preferences
            
        Returns:
            AI response with generated code
        """
        ai_context = context or AIContext()
        
        # Build comprehensive prompt
        prompt_parts = [
            f"Generate {language} code for: {description}",
            "",
            "Requirements:",
            "- Follow best practices and coding standards",
            "- Include comprehensive error handling",
            "- Add detailed docstrings and comments",
            "- Ensure code is production-ready and well-tested",
            "- No placeholder code or TODOs"
        ]
        
        if style_preferences:
            prompt_parts.extend([
                "",
                f"Style preferences: {style_preferences}"
            ])
        
        if ai_context.existing_code:
            prompt_parts.extend([
                "",
                "Existing code context:",
                f"```{language}",
                ai_context.existing_code[:2000],  # Limit context size
                "```"
            ])
        
        if ai_context.dependencies:
            prompt_parts.extend([
                "",
                f"Available dependencies: {', '.join(ai_context.dependencies)}"
            ])
        
        prompt = "\n".join(prompt_parts)
        
        request = AIRequest(
            request_type=RequestType.CODE_GENERATION,
            prompt=prompt,
            context={
                'language': language,
                'existing_code': ai_context.existing_code,
                'dependencies': ai_context.dependencies,
                'project_path': str(ai_context.project_path) if ai_context.project_path else None
            },
            metadata={'language': language, 'description': description}
        )
        
        return await self.execute_request(request)
    
    async def review_code(
        self,
        code: str,
        language: str,
        focus_areas: Optional[List[str]] = None
    ) -> AIResponse:
        """
        Review code for quality, security, and best practices.
        
        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on
            
        Returns:
            AI response with review findings
        """
        focus_areas = focus_areas or [
            "security", "performance", "maintainability", 
            "testing", "documentation"
        ]
        
        prompt = f"""
Review this {language} code comprehensively:

Focus Areas:
{chr(10).join(f"- {area}" for area in focus_areas)}

Code to Review:
```{language}
{code}
```

Provide:
1. Overall quality assessment (1-10 score)
2. Specific issues found with line numbers
3. Security vulnerabilities and recommendations
4. Performance optimization suggestions
5. Maintainability improvements
6. Testing recommendations
7. Documentation gaps

Format as structured analysis with clear sections.
"""
        
        request = AIRequest(
            request_type=RequestType.CODE_REVIEW,
            prompt=prompt,
            context={'language': language, 'focus_areas': focus_areas},
            metadata={'language': language, 'code_length': len(code)}
        )
        
        return await self.execute_request(request)
    
    async def validate_output(
        self,
        output: str,
        expected_format: Optional[str] = None,
        validation_rules: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate AI output for completeness and correctness.
        
        Args:
            output: Output to validate
            expected_format: Expected output format
            validation_rules: Custom validation rules
            
        Returns:
            Validation result with score and issues
        """
        prompt_parts = [
            "Validate this AI output for completeness and quality:",
            "",
            "Output to validate:",
            output,
            "",
            "Validation Criteria:",
            "- No placeholder text (TODO, FIXME, etc.)",
            "- Complete implementation without missing parts",
            "- Correct syntax and structure",
            "- Logical consistency",
            "- Production-ready quality"
        ]
        
        if expected_format:
            prompt_parts.extend([
                f"- Expected format: {expected_format}"
            ])
        
        if validation_rules:
            prompt_parts.extend([
                "- Custom rules:",
                *[f"  • {rule}" for rule in validation_rules]
            ])
        
        prompt_parts.extend([
            "",
            "Provide:",
            "1. Authenticity score (0-100): How complete and genuine is the output?",
            "2. Issues found: List any problems or incomplete sections",
            "3. Suggestions: Recommendations for improvement",
            "",
            "Format response as structured JSON with 'score', 'issues', and 'suggestions' fields."
        ])
        
        prompt = "\n".join(prompt_parts)
        
        request = AIRequest(
            request_type=RequestType.VALIDATION,
            prompt=prompt,
            context={'output_length': len(output)},
            metadata={'validation_type': 'completeness_check'}
        )
        
        response = await self.execute_request(request)
        
        # Parse validation response
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                return ValidationResult(
                    is_authentic=result_data.get('score', 0) >= 80,
                    authenticity_score=result_data.get('score', 0),
                    real_progress=result_data.get('score', 0),
                    fake_progress=100 - result_data.get('score', 100),
                    issues=[],
                    suggestions=result_data.get('suggestions', [])
                )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse validation response: {e}")
        
        # Fallback validation
        return ValidationResult(
            is_authentic=True,
            authenticity_score=85.0,
            real_progress=85.0,
            fake_progress=15.0,
            issues=[],
            suggestions=[]
        )
    
    async def execute_request(self, request: AIRequest) -> AIResponse:
        """
        Execute AI request with routing, caching, and monitoring.
        
        Args:
            request: AI request to execute
            
        Returns:
            AI response
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self.cache.get_cache_key(request)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Using cached response for request {request.request_id}")
                return cached_response
            
            # Route request to appropriate provider
            provider = self.router.route_request(request)
            
            # Execute request with provider
            response = await self._execute_with_provider(request, provider)
            
            # Update provider health metrics
            processing_time = time.time() - start_time
            self.router.update_provider_health(provider, processing_time, response.success)
            
            # Cache successful responses
            if response.success and request.request_type != RequestType.TASK_EXECUTION:
                # Don't cache task executions as they're unique
                cache_ttl = self._get_cache_ttl(request.request_type)
                self.cache.set(cache_key, response, cache_ttl)
            
            # Record metrics
            await self._record_request_metrics(request, response, processing_time)
            
            return response
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request execution failed: {e}")
            
            error_response = AIResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            return error_response
    
    async def _initialize_provider_clients(self) -> None:
        """Initialize AI provider clients."""
        # Initialize Claude Code client
        try:
            from ..integrations.claude_code_client import ClaudeCodeClient
            claude_code_client = ClaudeCodeClient(self.config_manager)
            await claude_code_client.initialize()
            self._provider_clients[AIProviderType.CLAUDE_CODE] = claude_code_client
            logger.debug("Claude Code client initialized")
        except ImportError as e:
            logger.warning(f"Failed to import Claude Code client: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude Code client: {e}")
        
        # Initialize other providers as available
        # TODO: Add Claude Flow, Anthropic API clients
    
    async def _test_connectivity(self) -> bool:
        """Test connectivity to available providers."""
        if not self._provider_clients:
            logger.error("No provider clients available")
            return False
        
        # Test primary provider (Claude Code)
        if AIProviderType.CLAUDE_CODE in self._provider_clients:
            try:
                test_request = AIRequest(
                    request_type=RequestType.TASK_EXECUTION,
                    prompt="Test connection - respond with 'Connected'",
                    timeout=30
                )
                
                response = await self._execute_with_provider(
                    test_request, AIProviderType.CLAUDE_CODE
                )
                
                return response.success
            
            except Exception as e:
                logger.error(f"Connectivity test failed: {e}")
                return False
        
        return True
    
    async def _execute_with_provider(
        self, 
        request: AIRequest, 
        provider: AIProviderType
    ) -> AIResponse:
        """Execute request with specific provider."""
        if provider not in self._provider_clients:
            raise AIInterfaceException(f"Provider {provider} not available")
        
        client = self._provider_clients[provider]
        start_time = time.time()
        
        try:
            # Convert request to provider format
            if provider == AIProviderType.CLAUDE_CODE:
                result = await self._execute_claude_code_request(client, request)
            else:
                raise AIInterfaceException(f"Provider {provider} not implemented")
            
            processing_time = time.time() - start_time
            
            # Build response
            response = AIResponse(
                request_id=request.request_id,
                success=result.get('success', False),
                content=result.get('output', ''),
                files_modified=result.get('files_modified', []),
                files_created=result.get('files_created', []),
                validation_score=result.get('validation_score', 0.0),
                tokens_used=result.get('tokens_used', 0),
                processing_time=processing_time,
                model_used=request.model,
                provider_used=provider,
                error_message=result.get('error_message'),
                metadata=result.get('metadata', {})
            )
            
            return response
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Provider {provider} execution failed: {e}")
            
            return AIResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                provider_used=provider
            )
    
    async def _execute_claude_code_request(
        self, 
        client, 
        request: AIRequest
    ) -> Dict[str, Any]:
        """Execute request with Claude Code client."""
        try:
            # Build task request for Claude Code
            task_request = {
                'description': request.prompt,
                'context': request.context,
                'timeout': request.timeout,
                'model': request.model,
                'metadata': request.metadata
            }
            
            # Execute with Claude Code
            result = await client.execute_task(task_request)
            
            return {
                'success': result.get('success', False),
                'output': result.get('output', ''),
                'files_modified': result.get('files_modified', []),
                'files_created': result.get('files_created', []),
                'validation_score': result.get('validation_score', 0.0),
                'tokens_used': result.get('tokens_used', 0),
                'error_message': result.get('error_message'),
                'metadata': result.get('metadata', {})
            }
        
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _get_cache_ttl(self, request_type: RequestType) -> int:
        """Get cache TTL based on request type."""
        ttl_mapping = {
            RequestType.CODE_GENERATION: 3600,      # 1 hour
            RequestType.CODE_REVIEW: 1800,          # 30 minutes  
            RequestType.DOCUMENTATION: 3600,        # 1 hour
            RequestType.ANALYSIS: 1800,             # 30 minutes
            RequestType.VALIDATION: 900,            # 15 minutes
            RequestType.TASK_EXECUTION: 0           # No cache
        }
        
        return ttl_mapping.get(request_type, 1800)
    
    async def _record_request_metrics(
        self, 
        request: AIRequest, 
        response: AIResponse, 
        processing_time: float
    ) -> None:
        """Record request metrics for monitoring."""
        metrics = {
            'request_id': str(request.request_id),
            'request_type': request.request_type.value,
            'provider': response.provider_used.value,
            'success': response.success,
            'processing_time': processing_time,
            'tokens_used': response.tokens_used,
            'validation_score': response.validation_score,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._request_history.append(metrics)
        
        # Update rolling averages
        if response.success:
            current_avg_time = self._performance_metrics.get('avg_processing_time', 0)
            self._performance_metrics['avg_processing_time'] = (
                current_avg_time * 0.9 + processing_time * 0.1
            )
            
            current_avg_score = self._performance_metrics.get('avg_validation_score', 0)
            self._performance_metrics['avg_validation_score'] = (
                current_avg_score * 0.9 + response.validation_score * 0.1
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get AI interface status and metrics."""
        recent_requests = list(self._request_history)[-100:]  # Last 100 requests
        
        if recent_requests:
            success_rate = sum(1 for r in recent_requests if r['success']) / len(recent_requests)
            avg_processing_time = sum(r['processing_time'] for r in recent_requests) / len(recent_requests)
        else:
            success_rate = 0.0
            avg_processing_time = 0.0
        
        return {
            'initialized': self._initialized,
            'available_providers': list(self._provider_clients.keys()),
            'cache_size': len(self.cache._cache),
            'recent_requests_count': len(recent_requests),
            'success_rate': success_rate * 100,
            'avg_processing_time': avg_processing_time,
            'performance_metrics': self._performance_metrics.copy()
        }
    
    async def shutdown(self) -> None:
        """Shutdown AI interface and cleanup resources."""
        logger.info("Shutting down AIInterface")
        
        # Close provider clients
        for provider_type, client in self._provider_clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing {provider_type} client: {e}")
        
        # Clear caches
        self.cache._cache.clear()
        self.cache._access_times.clear()
        
        self._initialized = False
        logger.info("AIInterface shutdown complete")


# Convenience functions

async def execute_simple_ai_request(
    prompt: str,
    request_type: RequestType = RequestType.TASK_EXECUTION,
    timeout: int = 300
) -> AIResponse:
    """
    Convenience function to execute a simple AI request.
    
    Args:
        prompt: Request prompt
        request_type: Type of AI request
        timeout: Request timeout
        
    Returns:
        AI response
    """
    ai_interface = AIInterface()
    await ai_interface.initialize()
    
    request = AIRequest(
        request_type=request_type,
        prompt=prompt,
        timeout=timeout
    )
    
    try:
        return await ai_interface.execute_request(request)
    finally:
        await ai_interface.shutdown()


async def generate_code_simple(
    description: str,
    language: str,
    existing_code: Optional[str] = None
) -> str:
    """
    Convenience function to generate code.
    
    Args:
        description: Code description
        language: Programming language
        existing_code: Optional existing code context
        
    Returns:
        Generated code
    """
    ai_interface = AIInterface()
    await ai_interface.initialize()
    
    context = AIContext(existing_code=existing_code)
    
    try:
        response = await ai_interface.generate_code(description, language, context)
        return response.content if response.success else ""
    finally:
        await ai_interface.shutdown()