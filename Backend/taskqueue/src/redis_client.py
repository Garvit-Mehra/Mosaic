"""Async Redis connection pool management."""

import redis.asyncio as redis

from src.config import settings

# Redis connection pool (initialized on app startup)
_redis_pool: redis.ConnectionPool | None = None
_redis_client: redis.Redis | None = None


def get_redis_pool() -> redis.ConnectionPool:
    """Get or create the Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            decode_responses=True,
        )
    return _redis_pool


def get_redis_client() -> redis.Redis:
    """Get the shared async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(connection_pool=get_redis_pool())
    return _redis_client


async def init_redis() -> None:
    """Initialize Redis connection and verify connectivity (called on app startup)."""
    client = get_redis_client()
    await client.ping()


async def close_redis() -> None:
    """Close Redis connections (called on app shutdown)."""
    global _redis_client, _redis_pool
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None


async def check_redis_health() -> bool:
    """Check if Redis is reachable. Returns True if healthy, False otherwise."""
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError, OSError):
        return False
