import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.excepthook import ExcepthookIntegration
import logging
import os

class SentryConfig:
    """Sentry configuration for comprehensive error tracking and monitoring"""
    
    def __init__(self):
        self.dsn = os.getenv('SENTRY_DSN')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.release = os.getenv('RELEASE_VERSION', '1.0.0')
        self.sample_rate = float(os.getenv('SENTRY_SAMPLE_RATE', '0.1'))
        self.traces_sample_rate = float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1'))
    
    def init_sentry(self):
        """Initialize Sentry with comprehensive integrations"""
        
        # Logging integration
        logging_integration = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )
        
        sentry_sdk.init(
            dsn=self.dsn,
            environment=self.environment,
            release=self.release,
            sample_rate=self.sample_rate,
            traces_sample_rate=self.traces_sample_rate,
            
            integrations=[
                FlaskIntegration(
                    transaction_style='endpoint'
                ),
                SqlalchemyIntegration(),
                RedisIntegration(),
                logging_integration,
                ExcepthookIntegration(always_run=True),
            ],
            
            # Performance monitoring
            enable_tracing=True,
            
            # Additional configuration
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send personal information
            
            # Custom error filtering
            before_send=self.before_send_filter,
            before_send_transaction=self.before_send_transaction_filter,
            
            # Debug mode
            debug=os.getenv('SENTRY_DEBUG', 'False').lower() == 'true'
        )
    
    def before_send_filter(self, event, hint):
        """Filter events before sending to Sentry"""
        
        # Skip certain exceptions
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            
            # Skip common expected errors
            if exc_type.__name__ in [
                'KeyboardInterrupt',
                'SystemExit',
                'BrokenPipeError'
            ]:
                return None
        
        # Add custom tags
        event.setdefault('tags', {}).update({
            'component': 'hive-mind',
            'service': 'claude-tui'
        })
        
        # Add user context (without PII)
        if 'user' not in event:
            event['user'] = {
                'id': 'anonymous',
                'ip_address': '{{auto}}'
            }
        
        return event
    
    def before_send_transaction_filter(self, event, hint):
        """Filter transaction events"""
        
        # Skip health check transactions
        if event.get('transaction', '').startswith('/health'):
            return None
            
        return event
    
    def add_breadcrumb(self, message, category='custom', level='info', data=None):
        """Add custom breadcrumb"""
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    
    def capture_exception(self, exception, **kwargs):
        """Capture exception with additional context"""
        return sentry_sdk.capture_exception(exception, **kwargs)
    
    def capture_message(self, message, level='info', **kwargs):
        """Capture custom message"""
        return sentry_sdk.capture_message(message, level=level, **kwargs)
    
    def set_user(self, user_info):
        """Set user context"""
        sentry_sdk.set_user(user_info)
    
    def set_tag(self, key, value):
        """Set custom tag"""
        sentry_sdk.set_tag(key, value)
    
    def set_extra(self, key, value):
        """Set extra data"""
        sentry_sdk.set_extra(key, value)

# Initialize global Sentry instance
sentry_config = SentryConfig()

# Middleware for Flask/FastAPI applications
class SentryMiddleware:
    """Middleware for enhanced Sentry integration"""
    
    def __init__(self, app):
        self.app = app
        sentry_config.init_sentry()
    
    def __call__(self, environ, start_response):
        # Add request context
        with sentry_sdk.push_scope() as scope:
            scope.set_tag('request_id', environ.get('HTTP_X_REQUEST_ID'))
            scope.set_extra('request_method', environ.get('REQUEST_METHOD'))
            scope.set_extra('request_path', environ.get('PATH_INFO'))
            
            return self.app(environ, start_response)

# Custom error handlers
def setup_error_handlers(app):
    """Setup custom error handlers with Sentry integration"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        sentry_config.capture_message(f"404 error: {error}", level='warning')
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        sentry_config.capture_exception(error.original_exception)
        return {"error": "Internal server error"}, 500
    
    @app.errorhandler(Exception)
    def unhandled_exception(error):
        sentry_config.capture_exception(error)
        return {"error": "An unexpected error occurred"}, 500

# Performance monitoring decorators
def monitor_performance(transaction_name=None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with sentry_sdk.start_transaction(
                op="function",
                name=transaction_name or func.__name__
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Hive Mind specific error tracking
class HiveMindErrorTracker:
    """Specialized error tracking for Hive Mind components"""
    
    @staticmethod
    def track_agent_error(agent_id, error, context=None):
        """Track agent-specific errors"""
        with sentry_sdk.push_scope() as scope:
            scope.set_tag('component', 'hive-agent')
            scope.set_tag('agent_id', agent_id)
            scope.set_extra('context', context or {})
            sentry_config.capture_exception(error)
    
    @staticmethod
    def track_coordination_error(coordination_type, error, agents_involved=None):
        """Track coordination errors"""
        with sentry_sdk.push_scope() as scope:
            scope.set_tag('component', 'coordination')
            scope.set_tag('coordination_type', coordination_type)
            scope.set_extra('agents_involved', agents_involved or [])
            sentry_config.capture_exception(error)
    
    @staticmethod
    def track_performance_issue(metric_name, value, threshold):
        """Track performance issues"""
        sentry_config.capture_message(
            f"Performance issue: {metric_name} = {value} (threshold: {threshold})",
            level='warning'
        )
        sentry_config.set_extra('metric_name', metric_name)
        sentry_config.set_extra('metric_value', value)
        sentry_config.set_extra('threshold', threshold)

# Export configuration
__all__ = [
    'SentryConfig',
    'sentry_config',
    'SentryMiddleware',
    'setup_error_handlers',
    'monitor_performance',
    'HiveMindErrorTracker'
]