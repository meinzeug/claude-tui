"""
Database dependencies for session management.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
import os
from typing import AsyncGenerator

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./claude_tui.db")

# Create async engine
async_engine = create_async_engine(
    DATABASE_URL,
    echo=bool(os.getenv("DATABASE_ECHO", "false").lower() == "true"),
    future=True
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=True,
    autocommit=False
)


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    This function creates a new database session for each request,
    ensures proper cleanup, and handles transactions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncSession:
    """
    Get database session for manual usage.
    
    Note: This should be used carefully as it doesn't auto-close.
    Prefer using get_database() dependency in FastAPI routes.
    """
    return AsyncSessionLocal()


async def close_db_connections():
    """Close all database connections."""
    await async_engine.dispose()


# Health check for database
async def check_database_health() -> bool:
    """Check if database is accessible."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception:
        return False