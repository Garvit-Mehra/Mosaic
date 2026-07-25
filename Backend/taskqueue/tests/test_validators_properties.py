"""Property-based tests for input validation rejection.

**Validates: Requirements 1.4, 1.5, 1.6, 1.7**

Uses Hypothesis to generate invalid job submissions and verify:
- Invalid job_type (not registered) is rejected with 400 (Req 1.4)
- Out-of-range priority/timeout/max_retries rejected with 400 (Req 1.5)
- Past execute_at is rejected with 400 (Req 1.6)
- Oversized payload is rejected with 413 (Req 1.7)
"""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.handler_registry import HandlerRegistry
from src.core.validators import (
    ValidationError,
    validate_execute_at,
    validate_job_submission,
    validate_job_type,
    validate_max_retries,
    validate_payload_size,
    validate_priority,
    validate_timeout_seconds,
)


# --- Helpers ---


def _make_registry(*job_types: str) -> HandlerRegistry:
    """Create a handler registry with the given job types registered."""
    registry = HandlerRegistry()
    for jt in job_types:
        registry.register(jt, lambda payload: None)
    return registry


# Known registered types used throughout the tests
REGISTERED_TYPES = ["email", "webhook", "data_export"]


def _valid_registry() -> HandlerRegistry:
    """Create a registry with standard test job types."""
    return _make_registry(*REGISTERED_TYPES)


# --- Strategies for invalid inputs ---

# Priorities outside [0, 10000]
invalid_priority_low = st.integers(max_value=-1)
invalid_priority_high = st.integers(min_value=10001)
invalid_priority_strategy = st.one_of(invalid_priority_low, invalid_priority_high)

# Timeouts outside [1, 86400]
invalid_timeout_low = st.integers(max_value=0)
invalid_timeout_high = st.integers(min_value=86401)
invalid_timeout_strategy = st.one_of(invalid_timeout_low, invalid_timeout_high)

# Max retries outside [0, 100]
invalid_retries_low = st.integers(max_value=-1)
invalid_retries_high = st.integers(min_value=101)
invalid_retries_strategy = st.one_of(invalid_retries_low, invalid_retries_high)

# Past execute_at: datetimes that are clearly in the past
past_datetime_strategy = st.builds(
    lambda minutes_ago: datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    st.integers(min_value=2, max_value=525600),  # 2 minutes to 1 year ago
)

# Unregistered job types: strings that are NOT in our registered set
unregistered_job_type_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda s: s not in REGISTERED_TYPES)

# Valid inputs (for use in combined tests where only one field is invalid)
valid_priority_strategy = st.integers(min_value=0, max_value=10000)
valid_timeout_strategy = st.integers(min_value=1, max_value=86400)
valid_retries_strategy = st.integers(min_value=0, max_value=100)


# --- Property Tests ---


class TestInputValidationRejection:
    """Property 13: Input Validation Rejection.

    **Validates: Requirements 1.4, 1.5, 1.6, 1.7**
    """

    @given(priority=invalid_priority_strategy)
    @settings(max_examples=200)
    def test_out_of_range_priority_rejected_with_400(self, priority: int):
        """Any priority outside [0, 10000] must be rejected with status 400.

        **Validates: Requirements 1.5**
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(priority)
        assert exc_info.value.status_code == 400

    @given(timeout=invalid_timeout_strategy)
    @settings(max_examples=200)
    def test_out_of_range_timeout_rejected_with_400(self, timeout: int):
        """Any timeout_seconds outside [1, 86400] must be rejected with status 400.

        **Validates: Requirements 1.5**
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_seconds(timeout)
        assert exc_info.value.status_code == 400

    @given(max_retries=invalid_retries_strategy)
    @settings(max_examples=200)
    def test_out_of_range_max_retries_rejected_with_400(self, max_retries: int):
        """Any max_retries outside [0, 100] must be rejected with status 400.

        **Validates: Requirements 1.5**
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_max_retries(max_retries)
        assert exc_info.value.status_code == 400

    @given(job_type=unregistered_job_type_strategy)
    @settings(max_examples=200)
    def test_unregistered_job_type_rejected_with_400(self, job_type: str):
        """Any job_type not in the handler registry must be rejected with status 400.

        **Validates: Requirements 1.4**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_type(job_type, registry=registry)
        assert exc_info.value.status_code == 400
        assert "not registered" in exc_info.value.detail

    @given(past_time=past_datetime_strategy)
    @settings(max_examples=200)
    def test_past_execute_at_rejected_with_400(self, past_time: datetime):
        """Any execute_at in the past must be rejected with status 400.

        **Validates: Requirements 1.6**
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_execute_at(past_time)
        assert exc_info.value.status_code == 400
        assert "future" in exc_info.value.detail

    @given(
        extra_size=st.integers(min_value=1, max_value=1000),
        max_size=st.integers(min_value=10, max_value=1000),
    )
    @settings(max_examples=200)
    def test_oversized_payload_rejected_with_413(self, extra_size: int, max_size: int):
        """Any payload exceeding the size limit must be rejected with status 413.

        **Validates: Requirements 1.7**
        """
        # Generate a payload that is guaranteed to exceed max_size when serialized
        # The string "x" * N serializes to N+2 bytes (with quotes) in JSON
        # A dict {"d": "x"*N} serializes to {"d":"x...x"} which is 6 + N bytes
        payload = {"d": "x" * (max_size + extra_size)}
        with pytest.raises(ValidationError) as exc_info:
            validate_payload_size(payload, max_size_bytes=max_size)
        assert exc_info.value.status_code == 413
        assert "exceeds" in exc_info.value.detail

    @given(
        priority=invalid_priority_strategy,
        timeout=valid_timeout_strategy,
        max_retries=valid_retries_strategy,
    )
    @settings(max_examples=200)
    def test_full_submission_rejects_invalid_priority(
        self, priority: int, timeout: int, max_retries: int
    ):
        """validate_job_submission rejects when priority is out of range.

        **Validates: Requirements 1.5**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={"key": "value"},
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
                registry=registry,
            )
        assert exc_info.value.status_code == 400

    @given(
        priority=valid_priority_strategy,
        timeout=invalid_timeout_strategy,
        max_retries=valid_retries_strategy,
    )
    @settings(max_examples=200)
    def test_full_submission_rejects_invalid_timeout(
        self, priority: int, timeout: int, max_retries: int
    ):
        """validate_job_submission rejects when timeout_seconds is out of range.

        **Validates: Requirements 1.5**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={"key": "value"},
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
                registry=registry,
            )
        assert exc_info.value.status_code == 400

    @given(
        priority=valid_priority_strategy,
        timeout=valid_timeout_strategy,
        max_retries=invalid_retries_strategy,
    )
    @settings(max_examples=200)
    def test_full_submission_rejects_invalid_max_retries(
        self, priority: int, timeout: int, max_retries: int
    ):
        """validate_job_submission rejects when max_retries is out of range.

        **Validates: Requirements 1.5**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={"key": "value"},
                priority=priority,
                timeout_seconds=timeout,
                max_retries=max_retries,
                registry=registry,
            )
        assert exc_info.value.status_code == 400

    @given(job_type=unregistered_job_type_strategy)
    @settings(max_examples=200)
    def test_full_submission_rejects_unregistered_job_type(self, job_type: str):
        """validate_job_submission rejects when job_type is not registered.

        **Validates: Requirements 1.4**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type=job_type,
                payload={"key": "value"},
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                registry=registry,
            )
        assert exc_info.value.status_code == 400

    @given(past_time=past_datetime_strategy)
    @settings(max_examples=200)
    def test_full_submission_rejects_past_execute_at(self, past_time: datetime):
        """validate_job_submission rejects when execute_at is in the past.

        **Validates: Requirements 1.6**
        """
        registry = _valid_registry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={"key": "value"},
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                execute_at=past_time,
                registry=registry,
            )
        assert exc_info.value.status_code == 400

    @given(extra_size=st.integers(min_value=1, max_value=500))
    @settings(max_examples=200)
    def test_full_submission_rejects_oversized_payload(self, extra_size: int):
        """validate_job_submission rejects when payload exceeds size limit.

        **Validates: Requirements 1.7**
        """
        registry = _valid_registry()
        # Use a small limit to avoid generating huge strings
        max_size = 100
        payload = {"d": "x" * (max_size + extra_size)}
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload=payload,
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                registry=registry,
                max_payload_size_bytes=max_size,
            )
        assert exc_info.value.status_code == 413
