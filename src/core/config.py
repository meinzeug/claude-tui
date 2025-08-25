"""
Core Configuration Management System with Security Best Practices
Provides secure configuration handling for the application
"""
import os
import secrets
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseSettings, validator, Field
from cryptography.fernet import Fernet
import json
import yaml
import logging

logger = logging.getLogger(__name__)


class SecurityConfig(BaseSettings):
    """Security-related configuration with secure defaults"""
    
    # JWT Configuration
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Password Security
    password_min_length: int = 8
    password_require_special_chars: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # Security Headers
    security_headers_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration with connection security"""
    
    database_url: str = "sqlite+aiosqlite:///./claude_tui.db"
    database_echo: bool = False
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Connection Security
    ssl_mode: str = "disable"  # For production: "require" or "verify-full"
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("Database URL cannot be empty")
        return v
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Main application configuration"""
    
    app_name: str = "Claude TUI API"
    app_version: str = "1.0.0"
    app_description: str = "Secure Claude TUI Backend API"
    debug: bool = False
    environment: str = "production"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError("Environment must be one of: development, staging, production")
        return v
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


class ConfigManager:
    """Centralized configuration manager with encryption support"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.cwd() / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize configurations
        self.app = AppConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        
        # Encryption key for sensitive data
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        logger.info(f"Configuration initialized for environment: {self.app.environment}")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive configuration data"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Restrict access
            logger.info("Generated new encryption key for configuration data")
            return key
    
    def encrypt_sensitive_value(self, value: str) -> str:
        """Encrypt sensitive configuration values"""
        return self._cipher.encrypt(value.encode()).decode()
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration values"""
        return self._cipher.decrypt(encrypted_value.encode()).decode()
    
    def load_from_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}
        
        try:
            content = config_file.read_text()
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(content) or {}
            elif config_file.suffix.lower() == '.json':
                return json.loads(content)
            else:
                logger.error(f"Unsupported configuration file format: {config_file.suffix}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            return {}
    
    def save_to_file(self, config_data: Dict[str, Any], config_file: Path) -> bool:
        """Save configuration to file"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                with open(config_file, 'w') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                logger.error(f"Unsupported configuration file format: {config_file.suffix}")
                return False
            
            # Set secure file permissions
            config_file.chmod(0o600)
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            return False
    
    def get_database_url(self) -> str:
        """Get database URL with proper security handling"""
        url = self.database.database_url
        
        # In production, ensure SSL is configured for remote databases
        if self.app.environment == "production" and not url.startswith("sqlite"):
            if "sslmode" not in url:
                logger.warning("Database URL missing SSL configuration in production")
        
        return url
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.app.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.app.environment == "production"
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.security.cors_origins,
            "allow_credentials": self.security.cors_allow_credentials,
            "allow_methods": self.security.cors_allow_methods,
            "allow_headers": self.security.cors_allow_headers,
        }
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers configuration"""
        if not self.security.security_headers_enabled:
            return {}
        
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": f"max-age={self.security.hsts_max_age}; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check JWT secret in production
        if self.is_production() and len(self.security.jwt_secret_key) < 64:
            warnings.append("JWT secret key should be at least 64 characters in production")
        
        # Check CORS configuration
        if self.is_production() and "*" in self.security.cors_origins:
            warnings.append("CORS should not allow all origins in production")
        
        # Check database configuration
        if self.is_production() and self.database.database_url.startswith("sqlite"):
            warnings.append("SQLite should not be used in production")
        
        # Check SSL configuration
        if self.is_production() and not self.database.ssl_mode in ["require", "verify-full"]:
            warnings.append("SSL should be enabled for database connections in production")
        
        return warnings


# Global configuration instance
config = ConfigManager()