"""Shared API dependencies for dependency injection of the coordinator."""

from fastapi import Depends, HTTPException, Request

from src.core.coordinator import JobCoordinator
from src.redis_client import check_redis_health


def get_coordinator(request: Request) -> JobCoordinator:
    """Get the JobCoordinator from the app state.

    The coordinator is initialized during application startup in the lifespan
    handler and stored in app.state.coordinator.

    Args:
        request: The current FastAPI request (provides access to app state).

    Returns:
        The singleton JobCoordinator instance.
    """
    return request.app.state.coordinator


async def require_redis() -> None:
    """Dependency that verifies Redis is reachable.

    Used on queue-dependent endpoints (POST /jobs, DELETE /jobs/:id) to return
    503 Service Unavailable when Redis is down. Read-only PostgreSQL-backed
    endpoints do not use this dependency and continue serving normally.

    Raises:
        HTTPException: 503 if Redis is unreachable.
    """
    healthy = await check_redis_health()
    if not healthy:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: queue backend is unreachable",
        )
