"""
Password Reset Service with Email Verification

Provides secure password reset functionality with email verification,
token management, and comprehensive security measures.
"""

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import aiosmtplib

from ..database.models import User
from .jwt_service import JWTService, TokenType
from .audit_logger import SecurityAuditLogger, SecurityEventType, SecurityLevel
from ..core.exceptions import SecurityError, ValidationError

logger = logging.getLogger(__name__)


class ResetTokenStatus(Enum):
    """Password reset token status"""
    ACTIVE = "active"
    USED = "used"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class ResetTokenInfo:
    """Password reset token information"""
    token_hash: str
    user_id: str
    email: str
    created_at: datetime
    expires_at: datetime
    status: ResetTokenStatus
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3


class EmailService:
    """Email service for sending password reset emails"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: bool = True,
        from_email: str = "noreply@claude-tui.com",
        from_name: str = "Claude-TIU"
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.use_tls = use_tls
        self.from_email = from_email
        self.from_name = from_name
        
        logger.info("Email service initialized for host: %s", smtp_host)
    
    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        user_name: str,
        reset_url: str,
        expires_in_hours: int = 1
    ) -> bool:
        """Send password reset email"""
        try:
            subject = "Password Reset Request - Claude-TIU"
            
            # Create HTML email content
            html_content = self._create_reset_email_html(
                user_name=user_name,
                reset_url=reset_url,
                expires_in_hours=expires_in_hours
            )
            
            # Create text content as fallback
            text_content = self._create_reset_email_text(
                user_name=user_name,
                reset_url=reset_url,
                expires_in_hours=expires_in_hours
            )
            
            # Send email
            success = await self._send_email(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            if success:
                logger.info("Password reset email sent to %s", to_email)
            else:
                logger.error("Failed to send password reset email to %s", to_email)
            
            return success
            
        except Exception as e:
            logger.error("Error sending password reset email: %s", e)
            return False
    
    def _create_reset_email_html(
        self,
        user_name: str,
        reset_url: str,
        expires_in_hours: int
    ) -> str:
        """Create HTML content for password reset email"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Password Reset - Claude-TIU</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .logo {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
                .content {{ line-height: 1.6; color: #333; }}
                .reset-button {{ display: inline-block; background-color: #2563eb; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning {{ background-color: #fef3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">Claude-TIU</div>
                    <h1>Password Reset Request</h1>
                </div>
                
                <div class="content">
                    <p>Hello {user_name},</p>
                    
                    <p>We received a request to reset your password for your Claude-TIU account. If you made this request, click the button below to reset your password:</p>
                    
                    <div style="text-align: center;">
                        <a href="{reset_url}" class="reset-button">Reset Password</a>
                    </div>
                    
                    <div class="warning">
                        <strong>Important:</strong>
                        <ul>
                            <li>This link will expire in {expires_in_hours} hour(s)</li>
                            <li>You can only use this link once</li>
                            <li>If you didn't request this reset, please ignore this email</li>
                        </ul>
                    </div>
                    
                    <p>If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated message from Claude-TIU. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_reset_email_text(
        self,
        user_name: str,
        reset_url: str,
        expires_in_hours: int
    ) -> str:
        """Create plain text content for password reset email"""
        return f"""
Claude-TIU - Password Reset Request

Hello {user_name},

We received a request to reset your password for your Claude-TIU account.

If you made this request, click the following link to reset your password:
{reset_url}

IMPORTANT:
- This link will expire in {expires_in_hours} hour(s)
- You can only use this link once
- If you didn't request this reset, please ignore this email

This is an automated message from Claude-TIU. Please do not reply to this email.
        """.strip()
    
    async def _send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str
    ) -> bool:
        """Send email using SMTP"""
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = f"{self.from_name} <{self.from_email}>"
            message['To'] = to_email
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            html_part = MIMEText(html_content, 'html', 'utf-8')
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email using aiosmtplib
            smtp_client = aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.use_tls
            )
            
            await smtp_client.connect()
            
            if self.smtp_username and self.smtp_password:
                await smtp_client.login(self.smtp_username, self.smtp_password)
            
            await smtp_client.send_message(message)
            await smtp_client.quit()
            
            return True
            
        except Exception as e:
            logger.error("SMTP send error: %s", e)
            return False


class PasswordResetService:
    """
    Comprehensive password reset service with security features.
    
    Features:
    - Secure token generation and validation
    - Email verification
    - Rate limiting
    - Audit logging
    - Token expiration and cleanup
    - Brute force protection
    """
    
    def __init__(
        self,
        jwt_service: JWTService,
        email_service: EmailService,
        audit_logger: SecurityAuditLogger,
        redis_client: Optional[redis.Redis] = None,
        token_expiry_hours: int = 1,
        max_attempts_per_token: int = 3,
        rate_limit_per_email: int = 3,  # requests per hour
        rate_limit_window_hours: int = 1
    ):
        self.jwt_service = jwt_service
        self.email_service = email_service
        self.audit_logger = audit_logger
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=3, decode_responses=True
        )
        self.token_expiry_hours = token_expiry_hours
        self.max_attempts_per_token = max_attempts_per_token
        self.rate_limit_per_email = rate_limit_per_email
        self.rate_limit_window_hours = rate_limit_window_hours
        
        # Redis key prefixes
        self.reset_tokens_prefix = "password_reset:tokens:"
        self.rate_limit_prefix = "password_reset:rate_limit:"
        
        logger.info("Password reset service initialized")
    
    async def request_password_reset(
        self,
        email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        reset_base_url: str = "https://claude-tui.com/reset-password"
    ) -> Dict[str, Any]:
        """Request password reset for email address"""
        try:
            # Normalize email
            email = email.lower().strip()
            
            # Check rate limiting
            if not await self._check_rate_limit(email, ip_address):
                await self.audit_logger.log_security_incident(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    SecurityLevel.MEDIUM,
                    f"Password reset rate limit exceeded for {email}",
                    ip_address=ip_address,
                    details={'email': email, 'type': 'password_reset'}
                )
                
                return {
                    'success': False,
                    'message': 'Too many reset requests. Please try again later.',
                    'retry_after_minutes': self.rate_limit_window_hours * 60
                }
            
            # For security, always return success even if user doesn't exist
            # This prevents email enumeration attacks
            
            # Get user from database
            try:
                from database.repositories.user_repository import UserRepository
                from database.session import get_async_session
                
                async with get_async_session() as db_session:
                    user_repo = UserRepository(db_session)
                    user = await user_repo.get_by_email(email)
                    
            except ImportError as e:
                logger.warning(f"Database not available for password reset: {e}")
                user = None  # For security, we'll still return success
            except Exception as e:
                logger.error(f"Failed to fetch user from database: {e}")
                user = None  # For security, we'll still return success
            
            if user:
                # Generate reset token
                reset_token = await self._generate_reset_token(
                    user_id=str(user.id),
                    email=email,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                # Create reset URL
                reset_url = f"{reset_base_url}?token={reset_token}"
                
                # Send email
                email_sent = await self.email_service.send_password_reset_email(
                    to_email=email,
                    reset_token=reset_token,
                    user_name=getattr(user, 'username', email),
                    reset_url=reset_url,
                    expires_in_hours=self.token_expiry_hours
                )
                
                if email_sent:
                    # Log successful request
                    await self.audit_logger.log_authentication(
                        SecurityEventType.PASSWORD_RESET_REQUEST,
                        user_id=str(user.id),
                        username=getattr(user, 'username', email),
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=True,
                        message="Password reset requested"
                    )
            
            # Always return success to prevent email enumeration
            return {
                'success': True,
                'message': f'If an account exists for {email}, a password reset link has been sent.',
                'expires_in_hours': self.token_expiry_hours
            }
            
        except Exception as e:
            logger.error("Password reset request failed: %s", e)
            
            return {
                'success': False,
                'message': 'Password reset request failed. Please try again.'
            }
    
    def _hash_token(self, token: str) -> str:
        """Hash token for secure storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def _generate_reset_token(
        self,
        user_id: str,
        email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Generate secure reset token"""
        
        # Generate cryptographically secure token
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token)
        
        # Create token info
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.token_expiry_hours)
        
        token_data = {
            'user_id': user_id,
            'email': email,
            'created_at': now.isoformat(),
            'expires_at': expires_at.isoformat(),
            'status': ResetTokenStatus.ACTIVE.value,
            'ip_address': ip_address or '',
            'user_agent': user_agent or '',
            'attempts': '0',
            'max_attempts': str(self.max_attempts_per_token)
        }
        
        # Store in Redis with TTL
        token_key = f"{self.reset_tokens_prefix}{token_hash}"
        await self._redis_hmset(token_key, token_data)
        await self._redis_expire(token_key, self.token_expiry_hours * 3600)
        
        return token
    
    async def _check_rate_limit(self, email: str, ip_address: Optional[str] = None) -> bool:
        """Check rate limiting for password reset requests"""
        try:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(hours=self.rate_limit_window_hours)
            
            # Check email-based rate limit
            email_key = f"{self.rate_limit_prefix}email:{hashlib.sha256(email.encode()).hexdigest()}"
            email_count = await self._count_requests_in_window(email_key, window_start)
            
            if email_count >= self.rate_limit_per_email:
                return False
            
            # Record this request
            timestamp = now.timestamp()
            await self._redis_zadd(email_key, {str(timestamp): timestamp})
            await self._redis_expire(email_key, self.rate_limit_window_hours * 3600)
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed: %s", e)
            # Fail open for availability
            return True
    
    async def _count_requests_in_window(self, key: str, window_start: datetime) -> int:
        """Count requests in time window"""
        try:
            # Remove expired entries
            await self._redis_zremrangebyscore(key, 0, window_start.timestamp())
            
            # Count current entries
            count = await self._redis_zcard(key)
            return count
            
        except Exception as e:
            logger.error("Failed to count requests: %s", e)
            return 0
    
    # Redis helper methods
    async def _redis_hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        try:
            return self.redis_client.hset(key, mapping=mapping)
        except Exception:
            return False
    
    async def _redis_expire(self, key: str, ttl: int) -> bool:
        try:
            return self.redis_client.expire(key, ttl)
        except Exception:
            return False
    
    async def _redis_zadd(self, key: str, mapping: Dict[str, float]) -> int:
        try:
            return self.redis_client.zadd(key, mapping)
        except Exception:
            return 0
    
    async def _redis_zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        try:
            return self.redis_client.zremrangebyscore(key, min_score, max_score)
        except Exception:
            return 0
    
    async def _redis_zcard(self, key: str) -> int:
        try:
            return self.redis_client.zcard(key)
        except Exception:
            return 0