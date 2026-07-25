"""Property-based tests for priority queue ordering and score correctness.

**Validates: Requirements 4.1, 4.2, 4.3**

Uses Hypothesis to generate jobs with varying priorities and timestamps and verify:
- Higher priority always produces a lower score (Req 4.1, 4.3)
- Same priority respects FIFO by enqueue time (Req 4.2, 4.3)
"""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.priority_queue import calculate_queue_score


# --- Strategies ---

# Priority values: non-negative integers (spec says >= 0, range 0-10000)
priority_strategy = st.integers(min_value=0, max_value=10000)

# Timestamps: realistic datetime range in UTC (2020-2030)
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1, tzinfo=timezone.utc),
    max_value=datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
)


# --- Property Tests ---


class TestPriorityOrdering:
    """Property 1: Priority Ordering.

    Higher priority values must always produce lower scores, ensuring that
    ZPOPMIN dequeues higher-priority jobs first.

    **Validates: Requirements 4.1, 4.3**
    """

    @given(
        priority_a=priority_strategy,
        priority_b=priority_strategy,
        timestamp=timestamp_strategy,
    )
    @settings(max_examples=500)
    def test_higher_priority_produces_lower_score(self, priority_a, priority_b, timestamp):
        """For the same timestamp, a higher priority must always yield a lower score.

        This guarantees that Redis ZPOPMIN returns higher-priority jobs first.

        **Validates: Requirements 4.1**
        """
        assume(priority_a != priority_b)

        score_a = calculate_queue_score(priority_a, timestamp)
        score_b = calculate_queue_score(priority_b, timestamp)

        if priority_a > priority_b:
            assert score_a < score_b, (
                f"Priority {priority_a} should produce lower score than {priority_b}, "
                f"but got {score_a} >= {score_b}"
            )
        else:
            assert score_a > score_b, (
                f"Priority {priority_b} should produce lower score than {priority_a}, "
                f"but got {score_b} >= {score_a}"
            )

    @given(
        priority_a=priority_strategy,
        priority_b=priority_strategy,
        timestamp_a=timestamp_strategy,
        timestamp_b=timestamp_strategy,
    )
    @settings(max_examples=500)
    def test_priority_dominates_timestamp(self, priority_a, priority_b, timestamp_a, timestamp_b):
        """A higher-priority job always has a lower score regardless of timestamps.

        The score formula uses -priority × 1_000_000_000_000 which creates gaps
        large enough that timestamp differences cannot override priority ordering.

        **Validates: Requirements 4.1, 4.3**
        """
        assume(priority_a > priority_b)

        score_a = calculate_queue_score(priority_a, timestamp_a)
        score_b = calculate_queue_score(priority_b, timestamp_b)

        assert score_a < score_b, (
            f"Higher priority ({priority_a}) should always produce lower score than "
            f"lower priority ({priority_b}), regardless of timestamps. "
            f"Got score_a={score_a}, score_b={score_b}"
        )


class TestQueueScoreCorrectness:
    """Property 2: Queue Score Correctness.

    The score formula must be exactly: -priority × 1,000,000,000,000 + enqueued_at_timestamp_ms.
    For same priority, earlier enqueue time must produce a lower score (FIFO).

    **Validates: Requirements 4.2, 4.3**
    """

    @given(priority=priority_strategy, timestamp=timestamp_strategy)
    @settings(max_examples=500)
    def test_score_matches_formula(self, priority, timestamp):
        """Score must equal -priority × 1_000_000_000_000 + enqueued_at_timestamp_ms.

        **Validates: Requirements 4.3**
        """
        timestamp_ms = int(timestamp.timestamp() * 1000)
        expected_score = float(-priority * 1_000_000_000_000 + timestamp_ms)

        actual_score = calculate_queue_score(priority, timestamp)

        assert actual_score == expected_score, (
            f"Score formula mismatch for priority={priority}, timestamp={timestamp}. "
            f"Expected {expected_score}, got {actual_score}"
        )

    @given(
        priority=priority_strategy,
        timestamp_a=timestamp_strategy,
        timestamp_b=timestamp_strategy,
    )
    @settings(max_examples=500)
    def test_same_priority_fifo_by_enqueue_time(self, priority, timestamp_a, timestamp_b):
        """With equal priority, earlier enqueue time produces a lower score (FIFO ordering).

        This ensures that among same-priority jobs, the one enqueued first is
        dequeued first by ZPOPMIN.

        **Validates: Requirements 4.2**
        """
        assume(timestamp_a != timestamp_b)

        score_a = calculate_queue_score(priority, timestamp_a)
        score_b = calculate_queue_score(priority, timestamp_b)

        if timestamp_a < timestamp_b:
            assert score_a < score_b, (
                f"Earlier timestamp {timestamp_a} should produce lower score "
                f"than {timestamp_b} at same priority {priority}. "
                f"Got score_a={score_a}, score_b={score_b}"
            )
        else:
            assert score_a > score_b, (
                f"Later timestamp {timestamp_a} should produce higher score "
                f"than {timestamp_b} at same priority {priority}. "
                f"Got score_a={score_a}, score_b={score_b}"
            )

    @given(priority=priority_strategy)
    @settings(max_examples=500)
    def test_zero_priority_score_equals_timestamp_ms(self, priority):
        """When priority is 0, score equals the timestamp in milliseconds.

        This validates the additive component of the formula works correctly.

        **Validates: Requirements 4.3**
        """
        # Use priority=0 to isolate the timestamp component
        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        timestamp_ms = int(timestamp.timestamp() * 1000)

        score = calculate_queue_score(0, timestamp)

        assert score == float(timestamp_ms), (
            f"With priority=0, score should equal timestamp_ms ({timestamp_ms}). "
            f"Got {score}"
        )

    @given(
        priority=priority_strategy,
        timestamp=timestamp_strategy,
    )
    @settings(max_examples=500)
    def test_score_decreases_by_1e12_per_priority_unit(self, priority, timestamp):
        """Each unit of priority decreases the score by exactly 1_000_000_000_000.

        This validates the multiplicative component of the formula.

        **Validates: Requirements 4.3**
        """
        assume(priority > 0)

        score_at_priority = calculate_queue_score(priority, timestamp)
        score_at_zero = calculate_queue_score(0, timestamp)

        # The difference should be priority * 1e12
        expected_diff = priority * 1_000_000_000_000
        actual_diff = score_at_zero - score_at_priority

        assert actual_diff == expected_diff, (
            f"Score difference between priority 0 and priority {priority} should be "
            f"{expected_diff}, got {actual_diff}"
        )
