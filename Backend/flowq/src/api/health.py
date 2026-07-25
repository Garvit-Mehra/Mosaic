"""PostgreSQL health check utilities.

Provides functions to check PostgreSQL connectivity for use in middleware
and dependency injection. When PG is unreachable, the API returns 503 for
ALL endpoints (Req 15.3).
"""

import logging

from sqlalchemy import text

from src.database import engine

logger = logging.getLogger(__name__)


async def check_pg_health() -> bool:
    """Check if PostgreSQL is reachable by executing a simple query.

    Attempts to execute 'SELECT 1' using the connection pool.
    Uses the configured postgres_connect_timeout for the attempt.

    Returns:
        True if PostgreSQL is healthy and responsive, False otherwise.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
        return False
