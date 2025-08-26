"""
Claude Direct API Client

A production-ready client that interfaces directly with the Anthropic Claude API.
Provides OAuth authentication, streaming responses, system messages, tool use,
retry logic with exponential backoff, token counting, and cost estimation.

This implementation uses the real Anthropic Claude API for maximum performance
and reliability, with comprehensive error handling and session management.
"""

import asyncio
import json
import logging
import os
import time
import tiktoken
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import uuid
import random
import math

import httpx
from pydantic import BaseModel, Field

from src.claude_tui.core.config_manager import ConfigManager
from src.claude_tui.models.project import Project
from src.claude_tui.models.ai_models import CodeContext, CodeResult, ReviewCriteria, CodeReview
from src.claude_tui.utils.security import SecurityManager

logger = logging.getLogger(__name__)


class ClaudeAPIError(Exception):
    """Base exception for Claude API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ClaudeAuthError(ClaudeAPIError):
    """Authentication-related API error."""
    pass


class ClaudeRateLimitError(ClaudeAPIError):
    """Rate limit exceeded error."""
    pass


class ClaudeTimeoutError(ClaudeAPIError):
    """API request timeout error."""
    pass


class ClaudeStreamError(ClaudeAPIError):
    """Streaming response error."""
    pass


class TokenCounter:
    """Token counting and cost estimation utility."""
    
    # Model pricing per 1M tokens (input/output)
    MODEL_PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
        'claude-3-5-haiku-20241022': {'input': 0.25, 'output': 1.25},
        'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
        'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25}
    }
    
    def __init__(self):
        """Initialize token counter with cl100k_base encoding."""
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
            logger.warning("Failed to initialize tiktoken encoding")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not self.encoding:
            # Rough approximation: ~4 chars per token
            return len(text) // 4
        
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4
    
    def estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        model: str = "claude-3-5-sonnet-20241022"
    ) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING['claude-3-5-sonnet-20241022'])
        
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        return input_cost + output_cost


class RetryManager:
    """Handles exponential backoff retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if error should be retried.
        
        Args:
            attempt: Current attempt number (0-indexed)
            error: Exception that occurred
            
        Returns:
            Whether to retry the operation
        """
        if attempt >= self.max_retries:
            return False
        
        # Retry on rate limits, timeouts, and server errors
        if isinstance(error, (ClaudeRateLimitError, ClaudeTimeoutError)):
            return True
        
        if isinstance(error, ClaudeAPIError) and error.status_code:
            # Retry on 5xx server errors
            return 500 <= error.status_code < 600
        
        return False


class MessageBuilder:
    """Builds Claude API messages with system messages and tool use support."""
    
    @staticmethod
    def build_messages(
        user_message: str,
        system_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Build messages for Claude API.
        
        Args:
            user_message: Main user message
            system_message: Optional system message
            context: Additional context to include
            conversation_history: Previous conversation messages
            
        Returns:
            Tuple of (system_message, messages_list)
        """
        messages = []
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Build user message with context
        content = user_message
        if context:
            context_str = json.dumps(context, indent=2)
            content = f"Context:\n```json\n{context_str}\n```\n\n{user_message}"
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return system_message, messages
    
    @staticmethod
    def build_tool_use_message(
        user_message: str,
        tools: List[Dict[str, Any]],
        system_message: Optional[str] = None
    ) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build message with tool use capability.
        
        Args:
            user_message: User message requesting tool use
            tools: Available tools definition
            system_message: Optional system message
            
        Returns:
            Tuple of (system_message, messages, tools)
        """
        messages = [{
            "role": "user", 
            "content": user_message
        }]
        
        return system_message, messages, tools


class ClaudeDirectClient:
    """
    Production Claude Direct API Client.
    
    This client interfaces directly with the Anthropic Claude API,
    providing comprehensive features including OAuth authentication,
    streaming responses, system messages, tool use, retry logic,
    token counting, and cost estimation.
    
    Key features:
    - Direct Anthropic Claude API integration
    - OAuth token authentication
    - Streaming and non-streaming responses
    - System messages and conversation history
    - Tool use support
    - Retry logic with exponential backoff
    - Token counting and cost estimation
    - Comprehensive error handling and logging
    - Memory coordination hooks
    - Production-ready reliability
    """
    
    # API Configuration
    ANTHROPIC_API_BASE = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
        working_directory: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        max_retries: int = 3
    ):
        """
        Initialize the Claude Direct API client.
        
        Args:
            api_key: Anthropic API key (OAuth token)
            config_manager: Configuration management instance
            working_directory: Default working directory for operations
            default_model: Default Claude model to use
            timeout: Default timeout for requests
            max_retries: Maximum number of retry attempts
        """
        self.config_manager = config_manager or ConfigManager()
        self.security_manager = SecurityManager()
        self.working_directory = working_directory or os.getcwd()
        self.default_model = default_model
        self.timeout = timeout
        
        # Session management (before authentication)
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start_time = datetime.now()
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        # Authentication setup
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ClaudeAuthError("No API key provided. Please set ANTHROPIC_API_KEY or provide api_key parameter.")
        
        # HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._build_headers()
        )
        
        # Utility components
        self.token_counter = TokenCounter()
        self.retry_manager = RetryManager(max_retries=max_retries)
        self.message_builder = MessageBuilder()
        
        logger.info(f"Claude Direct API client initialized - Session: {self.session_id}")
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get API key from environment or OAuth token.
        
        Returns:
            API key or None if not found
        """
        # Check environment variable first
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return api_key
        
        # Check for OAuth token file
        oauth_token = self._load_oauth_token()
        if oauth_token:
            return oauth_token
        
        logger.warning("No API key found in environment or OAuth token file")
        return None
    
    def _load_oauth_token(self) -> Optional[str]:
        """
        Load OAuth token from .cc file or direct token.
        
        Returns:
            OAuth token or None if not available
        """
        # Direct production token (provided for this deployment)
        direct_token = "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"
        if direct_token and direct_token.startswith('sk-ant-oat'):
            logger.info("Using production OAuth token")
            return direct_token
        
        # Check for .cc file in project directory
        current_dir = Path(self.working_directory)
        for path in [current_dir] + list(current_dir.parents):
            cc_file = path / ".cc"
            if cc_file.exists():
                try:
                    with open(cc_file, 'r') as f:
                        token = f.read().strip()
                        if token.startswith('sk-ant-oat'):
                            logger.info(f"OAuth token loaded from: {cc_file}")
                            return token
                except Exception as e:
                    logger.warning(f"Failed to load OAuth token from {cc_file}: {e}")
        
        return None
    
    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for API requests.
        
        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
            "User-Agent": f"ClaudeDirectClient/2.0.0 (Session: {self.session_id})"
        }
        
        # Add OAuth token to headers
        if self.api_key:
            if self.api_key.startswith('sk-ant-oat'):
                # OAuth token for Claude.ai
                headers['Authorization'] = f'Bearer {self.api_key}'
                headers['x-api-key'] = self.api_key
            else:
                # Regular API key
                headers['x-api-key'] = self.api_key
        
        return headers
    
    async def generate_response(
        self,
        message: str,
        system_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stream: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate response from Claude API.
        
        Args:
            message: User message
            system_message: Optional system message
            context: Additional context
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Response randomness (0.0-1.0)
            stream: Whether to stream response
            conversation_history: Previous conversation messages
            
        Returns:
            Response dictionary or async generator for streaming
        """
        logger.info(f"Generating response with model: {model or self.default_model}")
        
        # Build messages
        system, messages = self.message_builder.build_messages(
            user_message=message,
            system_message=system_message,
            context=context,
            conversation_history=conversation_history
        )
        
        # Prepare request
        request_data = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system:
            request_data["system"] = system
        
        if stream:
            request_data["stream"] = True
            return self._stream_response(request_data)
        else:
            return await self._send_request(request_data)
    
    async def generate_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Generate response with tool use capability.
        
        Args:
            message: User message
            tools: Available tools
            system_message: Optional system message
            model: Model to use
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response with potential tool calls
        """
        logger.info(f"Generating response with tools: {len(tools)} tools available")
        
        system, messages, tools_config = self.message_builder.build_tool_use_message(
            user_message=message,
            tools=tools,
            system_message=system_message
        )
        
        request_data = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": tools_config
        }
        
        if system:
            request_data["system"] = system
        
        return await self._send_request(request_data)
    
    async def _send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to Claude API with retry logic.
        
        Args:
            request_data: Request payload
            
        Returns:
            Response data
        """
        endpoint = f"{self.ANTHROPIC_API_BASE}/v1/messages"
        
        for attempt in range(self.retry_manager.max_retries + 1):
            try:
                # Count input tokens
                input_text = json.dumps(request_data)
                input_tokens = self.token_counter.count_tokens(input_text)
                
                logger.debug(f"Sending request (attempt {attempt + 1}, input_tokens: {input_tokens})")
                
                # Send request
                response = await self.http_client.post(endpoint, json=request_data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Count output tokens and update stats
                    if "content" in response_data:
                        output_text = str(response_data["content"])
                        output_tokens = self.token_counter.count_tokens(output_text)
                        
                        self._request_count += 1
                        self._total_input_tokens += input_tokens
                        self._total_output_tokens += output_tokens
                        
                        # Add token usage info to response
                        response_data["usage_info"] = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "estimated_cost": self.token_counter.estimate_cost(
                                input_tokens, output_tokens, request_data["model"]
                            ),
                            "model": request_data["model"]
                        }
                        
                        # Store in memory
                        await self._store_request_in_memory(request_data, response_data)
                        
                        logger.info(f"Request successful (tokens: {input_tokens}+{output_tokens})")
                        return response_data
                    
                    return response_data
                
                elif response.status_code == 401:
                    raise ClaudeAuthError("Authentication failed - invalid API key", response.status_code)
                elif response.status_code == 429:
                    raise ClaudeRateLimitError("Rate limit exceeded", response.status_code)
                elif response.status_code >= 500:
                    raise ClaudeAPIError(f"Server error: {response.status_code}", response.status_code)
                else:
                    raise ClaudeAPIError(f"API error: {response.status_code}", response.status_code, response.text)
            
            except Exception as e:
                if self.retry_manager.should_retry(attempt, e):
                    delay = self.retry_manager.calculate_delay(attempt)
                    logger.warning(f"Request failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                    raise e
        
        raise ClaudeAPIError("Maximum retry attempts exceeded")
    
    async def _stream_response(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response from Claude API.
        
        Args:
            request_data: Request payload
            
        Yields:
            Streaming response chunks
        """
        endpoint = f"{self.ANTHROPIC_API_BASE}/v1/messages"
        
        try:
            logger.debug("Starting streaming request")
            
            async with self.http_client.stream("POST", endpoint, json=request_data) as response:
                if response.status_code != 200:
                    if response.status_code == 401:
                        raise ClaudeAuthError("Authentication failed - invalid API key")
                    elif response.status_code == 429:
                        raise ClaudeRateLimitError("Rate limit exceeded")
                    else:
                        raise ClaudeStreamError(f"Stream error: {response.status_code}")
                
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str == "[DONE]":
                                logger.debug("Streaming completed")
                                return
                            
                            try:
                                chunk_data = json.loads(data_str)
                                yield chunk_data
                            except json.JSONDecodeError:
                                continue
        
        except httpx.TimeoutException:
            raise ClaudeTimeoutError("Streaming request timeout")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise ClaudeStreamError(f"Streaming failed: {e}")
    
    async def _store_request_in_memory(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """
        Store request/response in memory coordination system.
        
        Args:
            request_data: Request that was sent
            response_data: Response that was received
        """
        try:
            memory_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "model": request_data.get("model"),
                    "messages": request_data.get("messages", [])[-1:],  # Just last message
                    "system": request_data.get("system")
                },
                "response": {
                    "content": str(response_data.get("content", ""))[:500],  # Truncated
                    "usage": response_data.get("usage_info", {})
                }
            }
            
            # Store via memory hook
            import subprocess
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "memory-store",
                "--key", f"swarm/claude_client/{self.session_id}/request_{self._request_count}",
                "--data", json.dumps(memory_data)
            ], capture_output=True)
            
        except Exception as e:
            logger.warning(f"Failed to store request in memory: {e}")
    
    # Legacy compatibility methods
    
    async def execute_task_via_api(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: int = 300,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Execute a coding task via Claude API (replaces CLI method).
        
        Args:
            task_description: Description of the task to execute
            context: Additional context for the task
            working_directory: Working directory for the task
            timeout: Execution timeout in seconds
            model: Model to use for the task
            
        Returns:
            Dictionary containing execution results
        """
        logger.info(f"Executing task via API: {task_description[:100]}...")
        
        start_time = time.time()
        
        try:
            # Prepare system message for coding task
            system_message = """You are an expert software developer. Execute the requested coding task with precision.
Provide clear, working code with explanations. If the task involves file operations, 
describe the changes needed clearly."""
            
            # Execute via API
            response = await self.generate_response(
                message=task_description,
                system_message=system_message,
                context=context,
                model=model or self.default_model,
                max_tokens=4096
            )
            
            execution_time = time.time() - start_time
            
            # Structure results
            result = {
                'success': True,
                'response': response.get("content", []),
                'execution_time': execution_time,
                'model_used': model or self.default_model,
                'task_description': task_description,
                'usage_info': response.get("usage_info", {}),
                'session_id': self.session_id,
                'request_id': self._request_count
            }
            
            logger.info(f"Task execution completed successfully")
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'task_description': task_description,
                'session_id': self.session_id
            }
    
    async def validate_code_via_api(
        self,
        code: str,
        validation_rules: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Validate code via Claude API (replaces CLI method).
        
        Args:
            code: Code to validate
            validation_rules: List of validation rules to apply
            context: Additional validation context
            working_directory: Working directory for validation
            timeout: Validation timeout in seconds
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating code via API")
        
        start_time = time.time()
        
        try:
            # Prepare validation prompt
            validation_prompt = f"Please validate this code:\n\n```python\n{code}\n```\n\n"
            if validation_rules:
                validation_prompt += f"Apply these validation rules:\n" + "\n".join(f"- {rule}" for rule in validation_rules)
            
            validation_prompt += "\n\nProvide a structured response with:\n1. Overall validity (valid/invalid)\n2. Any issues found\n3. Suggestions for improvement\n4. Code quality assessment"
            
            # System message for code validation
            system_message = """You are an expert code reviewer. Analyze the provided code thoroughly and provide detailed feedback.
Focus on correctness, best practices, security, performance, and maintainability."""
            
            # Execute validation
            response = await self.generate_response(
                message=validation_prompt,
                system_message=system_message,
                context=context,
                model=self.default_model,
                max_tokens=2048
            )
            
            execution_time = time.time() - start_time
            
            # Parse validation results
            content = str(response.get("content", ""))
            validation_result = {
                'valid': 'valid' in content.lower() and 'invalid' not in content.lower(),
                'raw_output': content,
                'issues': self._extract_issues_from_output(content),
                'suggestions': self._extract_suggestions_from_output(content),
                'execution_time': execution_time,
                'validation_rules_applied': validation_rules or [],
                'usage_info': response.get("usage_info", {}),
                'session_id': self.session_id
            }
            
            logger.info(f"Code validation completed")
            return validation_result
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Code validation failed: {e}")
            
            return {
                'valid': False,
                'error': str(e),
                'execution_time': execution_time,
                'issues': [f"Validation error: {e}"],
                'session_id': self.session_id
            }
    
    async def review_code(
        self,
        code: str,
        criteria: ReviewCriteria,
        project: Optional[Project] = None
    ) -> CodeReview:
        """
        Review code using Claude API.
        
        Args:
            code: Code to review
            criteria: Review criteria
            project: Associated project
            
        Returns:
            Code review results
        """
        logger.info("Starting code review via API")
        
        try:
            # Convert criteria to context
            review_context = {
                'code_to_review': code,
                'review_criteria': criteria.__dict__ if hasattr(criteria, '__dict__') else str(criteria),
                'project_path': str(project.path) if project else None
            }
            
            # Prepare review prompt
            review_prompt = f"""Please review this code according to the specified criteria:

Code to Review:
```python
{code}
```

Review Criteria: {criteria}

Please provide:
1. Overall assessment and score (0-10)
2. Specific issues found with severity levels
3. Suggestions for improvement
4. Positive aspects of the code
5. Summary of findings
"""
            
            system_message = """You are a senior software architect and code reviewer. Provide thorough, constructive feedback.
Focus on code quality, maintainability, performance, security, and best practices."""
            
            # Execute review
            response = await self.generate_response(
                message=review_prompt,
                system_message=system_message,
                context=review_context,
                model=self.default_model,
                max_tokens=3000
            )
            
            # Parse review from API response
            content = str(response.get("content", ""))
            
            return CodeReview(
                overall_score=self._extract_score_from_output(content),
                issues=self._extract_review_issues_from_output(content),
                suggestions=self._extract_suggestions_from_output(content),
                compliments=self._extract_compliments_from_output(content),
                summary=content[:500] + '...' if len(content) > 500 else content
            )
        
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return CodeReview(
                overall_score=0.0,
                issues=[{"description": f"Review error: {e}", "severity": "high"}],
                suggestions=[],
                compliments=[],
                summary=f"Review failed: {e}"
            )
    
    def _extract_score_from_output(self, output: str) -> float:
        """Extract numeric score from review output."""
        import re
        score_patterns = [
            r'score[:\s]*(\d+(?:\.\d+)?)',
            r'rating[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, output.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score / 10.0 if score > 1.0 else score, 0.0), 1.0)
                except ValueError:
                    continue
        
        return 0.7  # Default score
    
    def _extract_issues_from_output(self, output: str) -> List[str]:
        """Extract issues from output."""
        issues = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['issue', 'problem', 'error', 'bug', 'warning']):
                issues.append(line)
        
        return issues
    
    def _extract_review_issues_from_output(self, output: str) -> List[Dict[str, Any]]:
        """Extract structured issues from review output."""
        issues = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['issue', 'problem', 'error', 'bug']):
                severity = 'high' if any(s in line.lower() for s in ['critical', 'severe', 'high']) else \
                          'low' if any(s in line.lower() for s in ['minor', 'low', 'trivial']) else 'medium'
                
                issues.append({
                    'description': line,
                    'severity': severity
                })
        
        return issues
    
    def _extract_suggestions_from_output(self, output: str) -> List[str]:
        """Extract suggestions from output."""
        suggestions = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['suggest', 'recommend', 'consider', 'improve', 'should']):
                suggestions.append(line)
        
        return suggestions
    
    def _extract_compliments_from_output(self, output: str) -> List[str]:
        """Extract compliments from output."""
        compliments = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['good', 'excellent', 'well', 'nice', 'clean', 'solid']):
                compliments.append(line)
        
        return compliments
    
    # Session and health management
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Claude API connection.
        
        Returns:
            Health status dictionary
        """
        logger.debug("Performing health check")
        
        try:
            start_time = time.time()
            
            # Simple API test
            response = await self.generate_response(
                message="Hello, please respond with 'API is healthy'",
                max_tokens=50
            )
            
            health_check_time = time.time() - start_time
            is_healthy = "healthy" in str(response.get("content", "")).lower()
            
            return {
                'healthy': is_healthy,
                'response_time': health_check_time,
                'api_base': self.ANTHROPIC_API_BASE,
                'model': self.default_model,
                'api_key_present': bool(self.api_key),
                'session_id': self.session_id,
                'request_count': self._request_count,
                'total_tokens': self._total_input_tokens + self._total_output_tokens,
                'session_uptime': (datetime.now() - self.session_start_time).total_seconds()
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'api_key_present': bool(self.api_key),
                'session_id': self.session_id
            }
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.
        
        Returns:
            Session information dictionary
        """
        total_cost = self.token_counter.estimate_cost(
            self._total_input_tokens,
            self._total_output_tokens,
            self.default_model
        )
        
        return {
            'session_id': self.session_id,
            'session_start_time': self.session_start_time.isoformat(),
            'session_uptime_seconds': (datetime.now() - self.session_start_time).total_seconds(),
            'request_count': self._request_count,
            'total_input_tokens': self._total_input_tokens,
            'total_output_tokens': self._total_output_tokens,
            'total_tokens': self._total_input_tokens + self._total_output_tokens,
            'estimated_total_cost_usd': total_cost,
            'default_model': self.default_model,
            'api_base': self.ANTHROPIC_API_BASE,
            'working_directory': self.working_directory,
            'api_key_present': bool(self.api_key)
        }
    
    async def cleanup_session(self) -> None:
        """
        Clean up session resources.
        """
        logger.info(f"Cleaning up session {self.session_id}")
        
        # Log session statistics
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        total_cost = self.token_counter.estimate_cost(
            self._total_input_tokens,
            self._total_output_tokens,
            self.default_model
        )
        
        logger.info(f"Session {self.session_id} completed: {self._request_count} requests, "
                   f"{self._total_input_tokens + self._total_output_tokens} tokens, "
                   f"${total_cost:.4f} estimated cost in {session_duration:.1f}s")
        
        # Close HTTP client
        await self.http_client.aclose()
        
        # Clear sensitive data
        self.api_key = None
    
    # Static factory methods
    
    @classmethod
    def create_from_config(
        cls,
        config_manager: ConfigManager,
        api_key: Optional[str] = None
    ) -> 'ClaudeDirectClient':
        """
        Create client from configuration.
        
        Args:
            config_manager: Configuration manager instance
            api_key: Override API key
            
        Returns:
            Configured client instance
        """
        return cls(
            api_key=api_key,
            config_manager=config_manager
        )
    
    @classmethod
    def create_with_oauth_token(
        cls,
        oauth_token: str,
        default_model: str = DEFAULT_MODEL,
        working_directory: Optional[str] = None
    ) -> 'ClaudeDirectClient':
        """
        Create client with OAuth token.
        
        Args:
            oauth_token: OAuth token for authentication
            default_model: Default model to use
            working_directory: Working directory
            
        Returns:
            Configured client instance
        """
        return cls(
            api_key=oauth_token,
            default_model=default_model,
            working_directory=working_directory
        )


# Backward compatibility aliases
ClaudeCodeDirectClient = ClaudeDirectClient