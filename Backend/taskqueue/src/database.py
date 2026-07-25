"""Async SQLAlchemy engine and session factory for PostgreSQL."""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings

# Create async engine with connection pool
engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "timeout": settings.postgres_connect_timeout,
    },
)

# Session factory for creating async sessions
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """Create and yield a new async database session.

    Usage as a dependency or context manager:
        async with async_session_factory() as session:
            ...
    """
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Initialize database connection pool (called on app startup)."""
    # Verify connectivity by attempting a connection
    async with engine.begin() as conn:
        await conn.run_sync(lambda _: None)


async def close_db() -> None:
    """Close database connection pool (called on app shutdown)."""
    await engine.dispose()
