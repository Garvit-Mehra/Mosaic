"""FastAPI application factory with lifespan handler."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import PostgresHealthMiddleware
from src.api.routes.dlq import router as dlq_router
from src.api.routes.jobs import router as jobs_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.workers import router as workers_router
from src.config import settings
from src.core.coordinator import JobCoordinator
from src.core.priority_queue import RedisPriorityQueue
from src.core.reconstruction import StateReconstructor
from src.database import async_session_factory, close_db, init_db
from src.redis_client import close_redis, get_redis_client, init_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting %s...", settings.app_name)

    try:
        await init_db()
        logger.info("PostgreSQL connection pool initialized")
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL: %s", e)
        raise

    try:
        await init_redis()
        logger.info("Redis connection pool initialized")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        raise

    # Initialize coordinator and store in app state
    redis_client = get_redis_client()
    priority_queue = RedisPriorityQueue(redis_client=redis_client)
    coordinator = JobCoordinator(
        session_factory=async_session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
    )
    app.state.coordinator = coordinator

    # Reconstruct Redis state from PostgreSQL (Req 13.2, 13.5)
    # Block new submissions until reconstruction completes
    reconstructor = StateReconstructor(
        session_factory=async_session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
    )
    app.state.reconstructor = reconstructor

    reconstruction_result = await reconstructor.reconstruct()
    logger.info(
        "State reconstruction complete: %d priority, %d scheduled, %d recovered (in %dms)",
        reconstruction_result["priority_queue_rebuilt"],
        reconstruction_result["scheduled_rebuilt"],
        reconstruction_result["running_recovered"],
        reconstruction_result["total_time_ms"],
    )

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.app_name)

    await close_redis()
    logger.info("Redis connections closed")

    await close_db()
    logger.info("PostgreSQL connections closed")

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Fault-tolerant distributed job queue and task scheduler",
        lifespan=lifespan,
    )

    # Register middleware (outermost = first to execute)
    app.add_middleware(PostgresHealthMiddleware)

    # Register routers
    app.include_router(jobs_router)
    app.include_router(dlq_router)
    app.include_router(workers_router)
    app.include_router(metrics_router)

    return app


# Application instance
app = create_app()
