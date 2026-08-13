"""Unit tests for job submission validation."""

from datetime import datetime, timedelta, timezone

import pytest

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


def _make_registry(*job_types: str) -> HandlerRegistry:
    """Create a handler registry with the given job types registered."""
    registry = HandlerRegistry()
    for jt in job_types:
        registry.register(jt, lambda payload: None)
    return registry


class TestValidatePriority:
    """Tests for priority validation."""

    def test_valid_priority_zero(self):
        validate_priority(0)  # Should not raise

    def test_valid_priority_max(self):
        validate_priority(10000)  # Should not raise

    def test_valid_priority_mid(self):
        validate_priority(5000)  # Should not raise

    def test_invalid_priority_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(-1)
        assert exc_info.value.status_code == 400
        assert "priority" in exc_info.value.detail

    def test_invalid_priority_too_high(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(10001)
        assert exc_info.value.status_code == 400
        assert "priority" in exc_info.value.detail


class TestValidateTimeoutSeconds:
    """Tests for timeout_seconds validation."""

    def test_valid_timeout_min(self):
        validate_timeout_seconds(1)  # Should not raise

    def test_valid_timeout_max(self):
        validate_timeout_seconds(86400)  # Should not raise

    def test_valid_timeout_mid(self):
        validate_timeout_seconds(300)  # Should not raise

    def test_invalid_timeout_zero(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_seconds(0)
        assert exc_info.value.status_code == 400
        assert "timeout_seconds" in exc_info.value.detail

    def test_invalid_timeout_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_seconds(-1)
        assert exc_info.value.status_code == 400

    def test_invalid_timeout_too_high(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_seconds(86401)
        assert exc_info.value.status_code == 400
        assert "timeout_seconds" in exc_info.value.detail


class TestValidateMaxRetries:
    """Tests for max_retries validation."""

    def test_valid_retries_zero(self):
        validate_max_retries(0)  # Should not raise

    def test_valid_retries_max(self):
        validate_max_retries(100)  # Should not raise

    def test_valid_retries_mid(self):
        validate_max_retries(50)  # Should not raise

    def test_invalid_retries_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_max_retries(-1)
        assert exc_info.value.status_code == 400
        assert "max_retries" in exc_info.value.detail

    def test_invalid_retries_too_high(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_max_retries(101)
        assert exc_info.value.status_code == 400
        assert "max_retries" in exc_info.value.detail


class TestValidateJobType:
    """Tests for job_type validation against handler registry."""

    def test_valid_registered_type(self):
        registry = _make_registry("email", "webhook")
        validate_job_type("email", registry=registry)  # Should not raise

    def test_invalid_unregistered_type(self):
        registry = _make_registry("email", "webhook")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_type("unknown_type", registry=registry)
        assert exc_info.value.status_code == 400
        assert "not registered" in exc_info.value.detail

    def test_invalid_type_shows_registered_types(self):
        registry = _make_registry("email", "webhook")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_type("unknown", registry=registry)
        assert "email" in exc_info.value.detail
        assert "webhook" in exc_info.value.detail

    def test_invalid_type_empty_registry(self):
        registry = HandlerRegistry()
        with pytest.raises(ValidationError) as exc_info:
            validate_job_type("any_type", registry=registry)
        assert exc_info.value.status_code == 400
        assert "not registered" in exc_info.value.detail


class TestValidateExecuteAt:
    """Tests for execute_at validation."""

    def test_none_is_valid(self):
        validate_execute_at(None)  # Should not raise

    def test_future_time_is_valid(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        validate_execute_at(future)  # Should not raise

    def test_past_time_is_invalid(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            validate_execute_at(past)
        assert exc_info.value.status_code == 400
        assert "future" in exc_info.value.detail

    def test_naive_past_time_is_invalid(self):
        """Naive datetime (no timezone) treated as UTC and validated."""
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            validate_execute_at(past)
        assert exc_info.value.status_code == 400

    def test_naive_future_time_is_valid(self):
        """Naive datetime (no timezone) treated as UTC and validated."""
        future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        validate_execute_at(future)  # Should not raise


class TestValidatePayloadSize:
    """Tests for payload size validation."""

    def test_small_payload_is_valid(self):
        validate_payload_size({"key": "value"})  # Should not raise

    def test_empty_payload_is_valid(self):
        validate_payload_size({})  # Should not raise

    def test_payload_at_limit_is_valid(self):
        # Create payload just under 100 bytes limit
        validate_payload_size({"a": "b"}, max_size_bytes=100)  # Should not raise

    def test_oversized_payload_returns_413(self):
        # Create a payload that exceeds a small limit
        large_payload = {"data": "x" * 200}
        with pytest.raises(ValidationError) as exc_info:
            validate_payload_size(large_payload, max_size_bytes=100)
        assert exc_info.value.status_code == 413
        assert "exceeds" in exc_info.value.detail

    def test_payload_exactly_at_limit(self):
        # Payload of exactly the limit size should be valid
        payload = {"k": "v"}
        import json
        size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        validate_payload_size(payload, max_size_bytes=size)  # Should not raise

    def test_payload_one_byte_over_limit(self):
        payload = {"k": "v"}
        import json
        size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        with pytest.raises(ValidationError) as exc_info:
            validate_payload_size(payload, max_size_bytes=size - 1)
        assert exc_info.value.status_code == 413

    def test_default_1mb_limit(self):
        # A small payload should be fine with default limit
        validate_payload_size({"data": "hello"})  # Should not raise


class TestValidateJobSubmission:
    """Tests for the combined validation function."""

    def test_valid_submission(self):
        registry = _make_registry("email")
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        validate_job_submission(
            job_type="email",
            payload={"to": "user@example.com"},
            priority=100,
            timeout_seconds=300,
            max_retries=3,
            execute_at=future,
            registry=registry,
        )  # Should not raise

    def test_valid_submission_no_schedule(self):
        registry = _make_registry("email")
        validate_job_submission(
            job_type="email",
            payload={"to": "user@example.com"},
            priority=0,
            timeout_seconds=60,
            max_retries=0,
            execute_at=None,
            registry=registry,
        )  # Should not raise

    def test_invalid_priority_stops_early(self):
        registry = _make_registry("email")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={},
                priority=-1,
                timeout_seconds=300,
                max_retries=3,
                registry=registry,
            )
        assert exc_info.value.status_code == 400
        assert "priority" in exc_info.value.detail

    def test_invalid_timeout_detected(self):
        registry = _make_registry("email")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={},
                priority=0,
                timeout_seconds=0,
                max_retries=3,
                registry=registry,
            )
        assert exc_info.value.status_code == 400
        assert "timeout_seconds" in exc_info.value.detail

    def test_invalid_max_retries_detected(self):
        registry = _make_registry("email")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={},
                priority=0,
                timeout_seconds=300,
                max_retries=101,
                registry=registry,
            )
        assert exc_info.value.status_code == 400
        assert "max_retries" in exc_info.value.detail

    def test_invalid_job_type_detected(self):
        registry = _make_registry("email")
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="nonexistent",
                payload={},
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                registry=registry,
            )
        assert exc_info.value.status_code == 400
        assert "not registered" in exc_info.value.detail

    def test_past_execute_at_detected(self):
        registry = _make_registry("email")
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload={},
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                execute_at=past,
                registry=registry,
            )
        assert exc_info.value.status_code == 400
        assert "future" in exc_info.value.detail

    def test_oversized_payload_detected(self):
        registry = _make_registry("email")
        large_payload = {"data": "x" * 2_000_000}
        with pytest.raises(ValidationError) as exc_info:
            validate_job_submission(
                job_type="email",
                payload=large_payload,
                priority=0,
                timeout_seconds=300,
                max_retries=3,
                registry=registry,
            )
        assert exc_info.value.status_code == 413
        assert "exceeds" in exc_info.value.detail
