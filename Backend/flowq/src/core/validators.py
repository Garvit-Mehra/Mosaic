"""Job submission validation logic.

Validates job fields against configured limits and registered handlers.
Raises appropriate HTTP exceptions for invalid inputs.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.config import settings
from src.core.handler_registry import HandlerRegistry, handler_registry


class ValidationError(Exception):
    """Raised when job validation fails.

    Attributes:
        status_code: HTTP status code to return (400 or 413).
        detail: Human-readable error message.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def validate_priority(priority: int) -> None:
    """Validate that priority is within the allowed range [0, 10000].

    Args:
        priority: The priority value to validate.

    Raises:
        ValidationError: If priority is outside the range 0-10000.
    """
    if priority < 0 or priority > settings.max_priority:
        raise ValidationError(
            status_code=400,
            detail=f"priority must be between 0 and {settings.max_priority}, got {priority}",
        )


def validate_timeout_seconds(timeout_seconds: int) -> None:
    """Validate that timeout_seconds is within the allowed range [1, 86400].

    Args:
        timeout_seconds: The timeout value in seconds to validate.

    Raises:
        ValidationError: If timeout_seconds is outside the range 1-86400.
    """
    if timeout_seconds < 1 or timeout_seconds > settings.max_job_timeout_seconds:
        raise ValidationError(
            status_code=400,
            detail=(
                f"timeout_seconds must be between 1 and {settings.max_job_timeout_seconds}, "
                f"got {timeout_seconds}"
            ),
        )


def validate_max_retries(max_retries: int) -> None:
    """Validate that max_retries is within the allowed range [0, 100].

    Args:
        max_retries: The max retries value to validate.

    Raises:
        ValidationError: If max_retries is outside the range 0-100.
    """
    if max_retries < 0 or max_retries > settings.max_retry_limit:
        raise ValidationError(
            status_code=400,
            detail=(
                f"max_retries must be between 0 and {settings.max_retry_limit}, "
                f"got {max_retries}"
            ),
        )


def validate_job_type(job_type: str, registry: Optional[HandlerRegistry] = None) -> None:
    """Validate that job_type is registered in the handler registry.

    Args:
        job_type: The job type string to validate.
        registry: Optional handler registry to check against. Defaults to global registry.

    Raises:
        ValidationError: If job_type is not registered.
    """
    reg = registry or handler_registry
    if not reg.is_registered(job_type):
        registered_types = reg.list_types()
        detail = f"job_type '{job_type}' is not registered."
        if registered_types:
            detail += f" Registered types: {registered_types}"
        raise ValidationError(status_code=400, detail=detail)


def validate_execute_at(execute_at: Optional[datetime]) -> None:
    """Validate that execute_at is in the future if provided.

    Args:
        execute_at: The scheduled execution time, or None for immediate execution.

    Raises:
        ValidationError: If execute_at is in the past.
    """
    if execute_at is None:
        return

    now = datetime.now(timezone.utc)
    # Make execute_at timezone-aware if it isn't already
    if execute_at.tzinfo is None:
        execute_at = execute_at.replace(tzinfo=timezone.utc)

    if execute_at <= now:
        raise ValidationError(
            status_code=400,
            detail="execute_at must be in the future",
        )


def validate_payload_size(
    payload: Any, max_size_bytes: Optional[int] = None
) -> None:
    """Validate that the JSON-serialized payload does not exceed the size limit.

    Args:
        payload: The job payload (dict or any JSON-serializable value).
        max_size_bytes: Optional override for the maximum payload size in bytes.
            Defaults to the configured payload_size_limit_bytes setting.

    Raises:
        ValidationError: If the serialized payload exceeds the size limit (413).
    """
    limit = max_size_bytes if max_size_bytes is not None else settings.payload_size_limit_bytes
    serialized = json.dumps(payload, separators=(",", ":"))
    size = len(serialized.encode("utf-8"))

    if size > limit:
        raise ValidationError(
            status_code=413,
            detail=(
                f"Payload size ({size} bytes) exceeds the allowed maximum "
                f"({limit} bytes)"
            ),
        )


def validate_job_submission(
    job_type: str,
    payload: Any,
    priority: int,
    timeout_seconds: int,
    max_retries: int,
    execute_at: Optional[datetime] = None,
    registry: Optional[HandlerRegistry] = None,
    max_payload_size_bytes: Optional[int] = None,
) -> None:
    """Run all validation checks for a job submission.

    Validates in order: priority, timeout_seconds, max_retries, job_type,
    execute_at, and payload size.

    Args:
        job_type: The job type string.
        payload: The job payload (JSON-serializable).
        priority: Job priority (0-10000).
        timeout_seconds: Job timeout in seconds (1-86400).
        max_retries: Maximum retry attempts (0-100).
        execute_at: Optional scheduled execution time.
        registry: Optional handler registry override.
        max_payload_size_bytes: Optional payload size limit override.

    Raises:
        ValidationError: If any validation check fails.
    """
    validate_priority(priority)
    validate_timeout_seconds(timeout_seconds)
    validate_max_retries(max_retries)
    validate_job_type(job_type, registry=registry)
    validate_execute_at(execute_at)
    validate_payload_size(payload, max_size_bytes=max_payload_size_bytes)
