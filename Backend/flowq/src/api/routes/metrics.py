"""Metrics API routes.

Provides an endpoint for system-level metrics including queue depth,
active workers, throughput, latency percentiles, and DLQ size.
"""

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_coordinator
from src.core.coordinator import JobCoordinator

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_metrics(
    request: Request,
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """Get current system metrics.

    Returns queue depth, active worker count, jobs-per-second throughput,
    P50 and P95 latency in milliseconds, and DLQ size.

    If no jobs have been processed, throughput and latency values are zero.

    Args:
        request: The current FastAPI request (provides access to app state).
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with queue_depth, active_workers, jobs_per_second,
        latency_p50_ms, latency_p95_ms, and dlq_size.
    """
    metrics = await coordinator.get_metrics()

    # If a metrics_collector is available in app state, overlay throughput/latency
    metrics_collector = getattr(request.app.state, "metrics_collector", None)
    if metrics_collector is not None:
        collector_data = metrics_collector.get_current_metrics()
        metrics["jobs_per_second"] = collector_data.get("jobs_per_second", 0.0)
        metrics["latency_p50_ms"] = collector_data.get("latency_p50_ms", 0.0)
        metrics["latency_p95_ms"] = collector_data.get("latency_p95_ms", 0.0)

    return metrics
