"""API middleware for infrastructure health checks.

Implements Req 15.3: When PostgreSQL is unreachable, ALL endpoints
return 503 Service Unavailable.
"""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.health import check_pg_health

logger = logging.getLogger(__name__)

# Cache PG health status to avoid checking on every request.
# This prevents overwhelming PG with health checks under high traffic.
_pg_health_cache: dict = {
    "healthy": True,
    "last_check": 0.0,
    "check_interval": 2.0,  # seconds between health checks
}


async def _is_pg_healthy() -> bool:
    """Check PG health with caching to avoid excessive queries.

    Caches the health status for `check_interval` seconds.
    When PG is down, checks more frequently (every 1s) to detect recovery quickly.
    """
    now = time.time()
    cache = _pg_health_cache

    # Use cached result if still fresh
    interval = cache["check_interval"] if cache["healthy"] else 1.0
    if now - cache["last_check"] < interval:
        return cache["healthy"]

    # Perform actual health check
    healthy = await check_pg_health()
    cache["healthy"] = healthy
    cache["last_check"] = now

    return healthy


def reset_pg_health_cache() -> None:
    """Reset the PG health cache. Useful for testing."""
    _pg_health_cache["healthy"] = True
    _pg_health_cache["last_check"] = 0.0


class PostgresHealthMiddleware(BaseHTTPMiddleware):
    """Middleware that returns 503 for ALL endpoints when PostgreSQL is down.

    Implements Req 15.3: IF PostgreSQL becomes unreachable, THEN the REST_API
    SHALL return 503 for all endpoints.

    Unlike Redis failures (which only block write operations), PostgreSQL
    failures block ALL operations since PG is the authoritative data store.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check PG health before processing any request.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            503 JSONResponse if PG is down, otherwise the normal response.
        """
        if not await _is_pg_healthy():
            logger.warning(
                f"PostgreSQL unavailable - returning 503 for {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Service Unavailable: PostgreSQL is unreachable"
                },
            )

        return await call_next(request)
