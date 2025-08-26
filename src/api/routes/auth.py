"""
Authentication routes for user login, registration, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import timedelta
import logging

from src.api.schemas.auth import (
    TokenResponse, LoginRequest, RegisterRequest,
    PasswordResetRequest, PasswordResetConfirm
)
from src.api.schemas.user import UserResponse
from src.api.models.user import User
from src.api.dependencies.database import get_database
from src.api.dependencies.auth import (
    authenticate_user, get_password_hash, create_user_token,
    get_user_by_username, get_user_by_email, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_database)
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    token_data = create_user_token(user)
    
    logger.info(f"User {user.username} logged in successfully")
    
    return TokenResponse(
        access_token=token_data["access_token"],
        token_type=token_data["token_type"],
        expires_in=token_data["expires_in"]
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_database)
):
    """
    Alternative login endpoint with JSON body.
    """
    user = await authenticate_user(login_data.username, login_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    token_data = create_user_token(user)
    
    logger.info(f"User {user.username} logged in via JSON endpoint")
    
    return TokenResponse(
        access_token=token_data["access_token"],
        token_type=token_data["token_type"],
        expires_in=token_data["expires_in"]
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: RegisterRequest,
    db: AsyncSession = Depends(get_database)
):
    """
    Register a new user account.
    """
    # Check if username already exists
    existing_user = await get_user_by_username(user_data.username, db)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await get_user_by_email(user_data.email, db)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_superuser=False
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {user_data.username}")
    
    return UserResponse.from_orm(new_user)


@router.post("/password-reset")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    db: AsyncSession = Depends(get_database)
):
    """
    Request password reset via email.
    """
    user = await get_user_by_email(reset_data.email, db)
    if not user:
        # Don't reveal whether email exists or not
        return {"message": "If the email exists, a reset link has been sent"}
    
    # In a real implementation, you would:
    # 1. Generate a secure reset token
    # 2. Store it with expiration time
    # 3. Send email with reset link
    
    logger.info(f"Password reset requested for user: {user.username}")
    
    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_database)
):
    """
    Confirm password reset with token.
    """
    # In a real implementation, you would:
    # 1. Validate the reset token
    # 2. Check if it's not expired
    # 3. Update the user's password
    # 4. Invalidate the token
    
    # For now, we'll return a placeholder response
    logger.info(f"Password reset confirmation attempted with token: {reset_data.token[:10]}...")
    
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password reset functionality not fully implemented"
    )


@router.post("/logout")
async def logout():
    """
    Logout endpoint (client should delete the token).
    """
    # Since JWTs are stateless, logout is handled client-side
    # In a production system, you might want to maintain a blacklist
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    """
    return UserResponse.from_orm(current_user)


# get_current_user imported above