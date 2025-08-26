"""
Additional authentication helper functions.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.user import User, UserSession


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Get user by ID from database."""
    try:
        stmt = select(User).where(User.id == user_id, User.is_active == True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    except Exception:
        return None


async def create_user_session(
    db: AsyncSession, 
    user_id: str, 
    token: str,
    expires_at: datetime,
    metadata: Dict = None
) -> UserSession:
    """Create a new user session."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    session = UserSession(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        metadata=metadata or {}
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return session


async def verify_user_session(db: AsyncSession, token: str) -> Optional[UserSession]:
    """Verify if user session exists and is valid."""
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        stmt = select(UserSession).where(
            UserSession.token_hash == token_hash,
            UserSession.expires_at > datetime.utcnow()
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
        
    except Exception:
        return None


async def revoke_user_session(db: AsyncSession, token: str):
    """Revoke a user session."""
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        stmt = select(UserSession).where(UserSession.token_hash == token_hash)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if session:
            await db.delete(session)
            await db.commit()
            
    except Exception:
        pass