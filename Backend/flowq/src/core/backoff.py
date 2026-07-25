"""Exponential backoff calculation for job retry delays.

Implements configurable exponential backoff with jitter to prevent
thundering herd problems when multiple jobs retry simultaneously.
"""

import random
from typing import Callable, Optional


def calculate_backoff(
    retry_count: int,
    base: float = 2.0,
    max_delay: float = 300.0,
    random_func: Optional[Callable[[float, float], float]] = None,
) -> float:
    """Calculate exponential backoff delay for a given retry attempt.

    Formula:
        raw_delay = base^retry_count
        capped_delay = min(raw_delay, max_delay)
        jitter = random value in [0, 0.1 * raw_delay]
        delay = capped_delay + jitter

    When raw_delay exceeds max_delay, the base delay is capped at max_delay
    but jitter is still added on top (based on the uncapped raw_delay when
    raw_delay <= max_delay, or based on max_delay when capped).

    Args:
        retry_count: The current retry attempt number (>= 0).
        base: The base for exponential calculation (>= 1.0 for monotonicity).
        max_delay: Maximum base delay in seconds (default 300.0).
        random_func: Optional function(low, high) -> float for deterministic
            testing. Defaults to random.uniform.

    Returns:
        The calculated backoff delay in seconds (always > 0 when base > 0).

    Raises:
        ValueError: If retry_count < 0, base <= 0, or max_delay <= 0.
    """
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    if base <= 0:
        raise ValueError("base must be positive")
    if max_delay <= 0:
        raise ValueError("max_delay must be positive")

    if random_func is None:
        random_func = random.uniform

    # Calculate raw exponential delay
    raw_delay = base ** retry_count

    # Cap the base delay at max_delay
    capped_delay = min(raw_delay, max_delay)

    # Jitter based on the raw (uncapped) delay per requirement 9.2:
    # jitter ∈ [0, 0.1 × base^retry_count]
    # But when capped, requirement 9.3 says "delay = cap + jitter"
    # so jitter is based on the capped value to keep total bounded
    jitter_base = capped_delay
    jitter = random_func(0.0, 0.1 * jitter_base)

    return capped_delay + jitter
