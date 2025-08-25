"""
AI Service for claude-tiu.

Manages AI integration and coordination across different AI providers:
- Claude Code integration
- Claude Flow coordination 
- Response validation and anti-hallucination
- Performance monitoring for AI operations
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..core.ai_interface import AIInterface
from ..core.exceptions import AIServiceError, ClaudeCodeError, ClaudeFlowError, ValidationError
from .base import BaseService


class AIService(BaseService):
    """
    AI Integration Management Service.
    
    Coordinates AI operations across Claude Code and Claude Flow,
    provides response validation and performance monitoring.
    """
    
    def __init__(self):
        super().__init__()
        self._ai_interface: Optional[AIInterface] = None
        self._claude_code_available = False
        self._claude_flow_available = False
        self._response_cache: Dict[str, Any] = {}
        self._request_history: List[Dict[str, Any]] = []
        
    async def _initialize_impl(self) -> None:
        """Initialize AI service with provider detection."""
        try:
            # Initialize AI Interface
            from ..core.ai_interface import AIInterface
            self._ai_interface = AIInterface()
            
            # Test Claude Code availability
            try:
                await self._test_claude_code_connection()
                self._claude_code_available = True
                self.logger.info("Claude Code integration available")
            except Exception as e:
                self.logger.warning(f"Claude Code not available: {str(e)}")
                self._claude_code_available = False
            
            # Test Claude Flow availability  
            try:
                await self._test_claude_flow_connection()
                self._claude_flow_available = True
                self.logger.info("Claude Flow integration available")
            except Exception as e:
                self.logger.warning(f"Claude Flow not available: {str(e)}")
                self._claude_flow_available = False
            
            if not self._claude_code_available and not self._claude_flow_available:
                raise AIServiceError("No AI providers available", service="ai_service")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize AI service: {str(e)}")
            raise AIServiceError(f"AI service initialization failed: {str(e)}", service="ai_service")
    
    async def _test_claude_code_connection(self) -> bool:
        """Test Claude Code connection."""
        # Simple test to verify Claude Code is responsive
        try:
            from ..integrations.claude_code import ClaudeCodeIntegration
            integration = ClaudeCodeIntegration()
            result = await integration.test_connection()
            return result.get('status') == 'connected'
        except ImportError:
            return False
        except Exception as e:
            self.logger.debug(f"Claude Code test failed: {str(e)}")
            return False
    
    async def _test_claude_flow_connection(self) -> bool:
        """Test Claude Flow connection."""
        try:
            from ..integrations.claude_flow import ClaudeFlowIntegration
            integration = ClaudeFlowIntegration()
            result = await integration.test_connection()
            return result.get('status') == 'connected'
        except ImportError:
            return False
        except Exception as e:
            self.logger.debug(f"Claude Flow test failed: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check with AI provider status."""
        base_health = await super().health_check()
        
        base_health.update({
            'claude_code_available': self._claude_code_available,
            'claude_flow_available': self._claude_flow_available,
            'response_cache_size': len(self._response_cache),
            'request_history_size': len(self._request_history)
        })
        
        return base_health
    
    async def generate_code(
        self,
        prompt: str,
        language: str = 'python',
        context: Optional[Dict[str, Any]] = None,
        validate_response: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code using AI with validation.
        
        Args:
            prompt: Code generation prompt
            language: Target programming language
            context: Additional context for generation
            validate_response: Whether to validate the response
            use_cache: Whether to use response caching
            
        Returns:
            Generated code with metadata
        """
        return await self.execute_with_monitoring(
            'generate_code',
            self._generate_code_impl,
            prompt=prompt,
            language=language,
            context=context,
            validate_response=validate_response,
            use_cache=use_cache
        )
    
    async def _generate_code_impl(
        self,
        prompt: str,
        language: str,
        context: Optional[Dict[str, Any]],
        validate_response: bool,
        use_cache: bool
    ) -> Dict[str, Any]:
        """Internal code generation implementation."""
        # Check cache first
        cache_key = self._create_cache_key('generate_code', prompt, language, context)
        if use_cache and cache_key in self._response_cache:
            self.logger.debug("Using cached response for code generation")
            return self._response_cache[cache_key]
        
        if not self._claude_code_available:
            raise ClaudeCodeError("Claude Code not available for code generation")
        
        try:
            # Prepare request
            request_data = {
                'type': 'code_generation',
                'prompt': prompt,
                'language': language,
                'context': context or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Generate code
            result = await self._ai_interface.generate_code(
                prompt=prompt,
                language=language,
                context=context or {}
            )
            
            # Validate response if requested
            if validate_response:
                validation_result = await self._validate_code_response(result, language)
                result['validation'] = validation_result
                
                if not validation_result['is_valid']:
                    self.logger.warning(
                        "Generated code failed validation",
                        extra={'validation_errors': validation_result['errors']}
                    )
            
            # Cache response
            if use_cache:
                self._response_cache[cache_key] = result
                
            # Store request history
            self._request_history.append({
                'request': request_data,
                'response': {
                    'success': True,
                    'code_length': len(result.get('code', '')),
                    'has_validation': 'validation' in result
                },
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            error_data = {
                'request': request_data,
                'response': {
                    'success': False,
                    'error': str(e)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            self._request_history.append(error_data)
            
            if isinstance(e, AIServiceError):
                raise
            else:
                raise ClaudeCodeError(f"Code generation failed: {str(e)}")
    
    async def orchestrate_task(
        self,
        task_description: str,
        requirements: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None,
        strategy: str = 'adaptive'
    ) -> Dict[str, Any]:
        """
        Orchestrate task using Claude Flow.
        
        Args:
            task_description: Description of task to orchestrate
            requirements: Task requirements and constraints
            agents: Specific agents to use (optional)
            strategy: Orchestration strategy
            
        Returns:
            Task orchestration result
        """
        return await self.execute_with_monitoring(
            'orchestrate_task',
            self._orchestrate_task_impl,
            task_description=task_description,
            requirements=requirements,
            agents=agents,
            strategy=strategy
        )
    
    async def _orchestrate_task_impl(
        self,
        task_description: str,
        requirements: Optional[Dict[str, Any]],
        agents: Optional[List[str]],
        strategy: str
    ) -> Dict[str, Any]:
        """Internal task orchestration implementation."""
        if not self._claude_flow_available:
            raise ClaudeFlowError("Claude Flow not available for task orchestration")
        
        try:
            from ..integrations.claude_flow import ClaudeFlowIntegration
            flow_integration = ClaudeFlowIntegration()
            
            # Prepare orchestration request
            orchestration_request = {
                'task': task_description,
                'requirements': requirements or {},
                'agents': agents or [],
                'strategy': strategy,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Execute orchestration
            result = await flow_integration.orchestrate_task(orchestration_request)
            
            # Store request history
            self._request_history.append({
                'request': orchestration_request,
                'response': {
                    'success': True,
                    'task_id': result.get('task_id'),
                    'agents_assigned': len(result.get('agents', []))
                },
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            error_data = {
                'request': orchestration_request,
                'response': {
                    'success': False,
                    'error': str(e)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            self._request_history.append(error_data)
            
            raise ClaudeFlowError(f"Task orchestration failed: {str(e)}")
    
    async def validate_response(
        self,
        response: Dict[str, Any],
        validation_type: str = 'general',
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate AI response for quality and correctness.
        
        Args:
            response: AI response to validate
            validation_type: Type of validation ('code', 'text', 'general')
            criteria: Validation criteria
            
        Returns:
            Validation result with score and issues
        """
        return await self.execute_with_monitoring(
            'validate_response',
            self._validate_response_impl,
            response=response,
            validation_type=validation_type,
            criteria=criteria
        )
    
    async def _validate_response_impl(
        self,
        response: Dict[str, Any],
        validation_type: str,
        criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal response validation implementation."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'errors': [],
            'warnings': [],
            'metadata': {
                'validation_type': validation_type,
                'criteria': criteria or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        try:
            if validation_type == 'code':
                return await self._validate_code_response(response, criteria.get('language', 'python'))
            elif validation_type == 'text':
                return await self._validate_text_response(response, criteria)
            else:
                return await self._validate_general_response(response, criteria)
                
        except Exception as e:
            self.logger.error(f"Response validation failed: {str(e)}")
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'errors': [f"Validation error: {str(e)}"]
            })
            return validation_result
    
    async def _validate_code_response(
        self,
        response: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """Validate code response for syntax and quality."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'errors': [],
            'warnings': [],
            'metadata': {
                'language': language,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        code = response.get('code', '')
        if not code or code.strip() == '':
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'errors': ['No code content found in response']
            })
            return validation_result
        
        # Check for common placeholder patterns
        placeholder_patterns = [
            '# TODO', '# FIXME', '# PLACEHOLDER', 
            'pass  # implement', '...',  # Python specific
            '// TODO', '// FIXME', '/* TODO',  # Multi-language
            'throw new NotImplementedException', 'NotImplementedError'
        ]
        
        placeholders_found = []
        for pattern in placeholder_patterns:
            if pattern.lower() in code.lower():
                placeholders_found.append(pattern)
        
        if placeholders_found:
            validation_result['warnings'].append(
                f"Found placeholder patterns: {', '.join(placeholders_found)}"
            )
            validation_result['score'] *= 0.7  # Reduce score for placeholders
        
        # Basic syntax validation for Python
        if language.lower() == 'python':
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                validation_result.update({
                    'is_valid': False,
                    'score': 0.0,
                    'errors': [f'Python syntax error: {str(e)}']
                })
        
        return validation_result
    
    async def _validate_text_response(
        self,
        response: Dict[str, Any],
        criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate text response for quality and completeness."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'errors': [],
            'warnings': [],
            'metadata': {
                'criteria': criteria or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        text = response.get('content', response.get('text', ''))
        if not text or text.strip() == '':
            validation_result.update({
                'is_valid': False,
                'score': 0.0,
                'errors': ['No text content found in response']
            })
            return validation_result
        
        # Check minimum length if specified
        if criteria and 'min_length' in criteria:
            min_length = criteria['min_length']
            if len(text.strip()) < min_length:
                validation_result['warnings'].append(
                    f"Text shorter than minimum length ({len(text)} < {min_length})"
                )
                validation_result['score'] *= 0.8
        
        return validation_result
    
    async def _validate_general_response(
        self,
        response: Dict[str, Any],
        criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate general response structure and content."""
        validation_result = {
            'is_valid': True,
            'score': 1.0,
            'errors': [],
            'warnings': [],
            'metadata': {
                'criteria': criteria or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Check required fields if specified
        if criteria and 'required_fields' in criteria:
            required_fields = criteria['required_fields']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                validation_result['errors'].append(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )
                validation_result['is_valid'] = False
                validation_result['score'] = 0.0
        
        return validation_result
    
    def _create_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Create cache key for response caching."""
        key_data = {
            'operation': operation,
            'args': args,
            'kwargs': kwargs
        }
        return f"{operation}:{hash(json.dumps(key_data, sort_keys=True, default=str))}"
    
    async def get_request_history(
        self,
        limit: Optional[int] = None,
        operation_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get AI request history with optional filtering."""
        history = self._request_history.copy()
        
        if operation_filter:
            history = [
                req for req in history 
                if req.get('request', {}).get('type') == operation_filter
            ]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        self.logger.info("AI service response cache cleared")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get AI service performance metrics."""
        total_requests = len(self._request_history)
        successful_requests = sum(
            1 for req in self._request_history 
            if req.get('response', {}).get('success', False)
        )
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'cache_size': len(self._response_cache),
            'claude_code_available': self._claude_code_available,
            'claude_flow_available': self._claude_flow_available,
            'providers_count': sum([self._claude_code_available, self._claude_flow_available])
        }