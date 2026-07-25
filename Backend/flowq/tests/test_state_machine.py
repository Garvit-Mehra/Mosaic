"""Unit tests for the job state machine."""

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from src.core.state_machine import (
    InvalidTransitionError,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    apply_transition,
    validate_transition,
)
from src.models.enums import JobStatus


@dataclass
class FakeJob:
    """Lightweight job stand-in for testing without DB dependencies."""

    status: JobStatus = JobStatus.PENDING
    updated_at: datetime = field(default_factory=datetime.utcnow)


class TestValidateTransition:
    """Tests for validate_transition function."""

    @pytest.mark.parametrize(
        "current,target",
        [
            (JobStatus.PENDING, JobStatus.QUEUED),
            (JobStatus.PENDING, JobStatus.SCHEDULED),
            (JobStatus.PENDING, JobStatus.CANCELLED),
            (JobStatus.SCHEDULED, JobStatus.QUEUED),
            (JobStatus.SCHEDULED, JobStatus.CANCELLED),
            (JobStatus.QUEUED, JobStatus.RUNNING),
            (JobStatus.QUEUED, JobStatus.CANCELLED),
            (JobStatus.RUNNING, JobStatus.COMPLETED),
            (JobStatus.RUNNING, JobStatus.FAILED),
            (JobStatus.FAILED, JobStatus.QUEUED),
            (JobStatus.FAILED, JobStatus.DEAD_LETTER),
        ],
    )
    def test_valid_transitions_return_true(self, current, target):
        assert validate_transition(current, target) is True

    @pytest.mark.parametrize(
        "current,target",
        [
            (JobStatus.PENDING, JobStatus.RUNNING),
            (JobStatus.PENDING, JobStatus.COMPLETED),
            (JobStatus.PENDING, JobStatus.FAILED),
            (JobStatus.SCHEDULED, JobStatus.RUNNING),
            (JobStatus.QUEUED, JobStatus.COMPLETED),
            (JobStatus.RUNNING, JobStatus.QUEUED),
            (JobStatus.RUNNING, JobStatus.CANCELLED),
            (JobStatus.FAILED, JobStatus.RUNNING),
            (JobStatus.FAILED, JobStatus.COMPLETED),
        ],
    )
    def test_invalid_transitions_return_false(self, current, target):
        assert validate_transition(current, target) is False

    @pytest.mark.parametrize("terminal", list(TERMINAL_STATES))
    def test_terminal_states_reject_all_targets(self, terminal):
        for target in JobStatus:
            assert validate_transition(terminal, target) is False


class TestApplyTransition:
    """Tests for apply_transition function."""

    def test_valid_transition_updates_status_and_timestamp(self):
        job = FakeJob(status=JobStatus.PENDING)
        old_updated = job.updated_at

        apply_transition(job, JobStatus.QUEUED)

        assert job.status == JobStatus.QUEUED
        assert job.updated_at >= old_updated

    def test_invalid_transition_raises_error_and_leaves_job_unchanged(self):
        job = FakeJob(status=JobStatus.PENDING)
        original_status = job.status
        original_updated = job.updated_at

        with pytest.raises(InvalidTransitionError) as exc_info:
            apply_transition(job, JobStatus.COMPLETED)

        assert job.status == original_status
        assert job.updated_at == original_updated
        assert "PENDING" in str(exc_info.value)
        assert "COMPLETED" in str(exc_info.value)

    @pytest.mark.parametrize("terminal", list(TERMINAL_STATES))
    def test_terminal_state_rejects_all_transitions(self, terminal):
        job = FakeJob(status=terminal)
        original_updated = job.updated_at

        for target in JobStatus:
            with pytest.raises(InvalidTransitionError) as exc_info:
                apply_transition(job, target)

            assert job.status == terminal
            assert job.updated_at == original_updated
            assert "terminal state" in str(exc_info.value)

    def test_error_message_includes_current_and_target(self):
        job = FakeJob(status=JobStatus.QUEUED)

        with pytest.raises(InvalidTransitionError) as exc_info:
            apply_transition(job, JobStatus.COMPLETED)

        error = exc_info.value
        assert error.current_status == JobStatus.QUEUED
        assert error.target_status == JobStatus.COMPLETED

    def test_full_happy_path_lifecycle(self):
        """A job can go through PENDING -> QUEUED -> RUNNING -> COMPLETED."""
        job = FakeJob(status=JobStatus.PENDING)

        apply_transition(job, JobStatus.QUEUED)
        assert job.status == JobStatus.QUEUED

        apply_transition(job, JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

        apply_transition(job, JobStatus.COMPLETED)
        assert job.status == JobStatus.COMPLETED

    def test_retry_path_lifecycle(self):
        """A job can go through PENDING -> QUEUED -> RUNNING -> FAILED -> QUEUED."""
        job = FakeJob(status=JobStatus.PENDING)

        apply_transition(job, JobStatus.QUEUED)
        apply_transition(job, JobStatus.RUNNING)
        apply_transition(job, JobStatus.FAILED)
        assert job.status == JobStatus.FAILED

        apply_transition(job, JobStatus.QUEUED)
        assert job.status == JobStatus.QUEUED

    def test_dead_letter_path(self):
        """A job can go FAILED -> DEAD_LETTER."""
        job = FakeJob(status=JobStatus.FAILED)

        apply_transition(job, JobStatus.DEAD_LETTER)
        assert job.status == JobStatus.DEAD_LETTER


class TestTransitionsMapCompleteness:
    """Ensure the transitions map covers all statuses."""

    def test_all_statuses_have_entries(self):
        for status in JobStatus:
            assert status in VALID_TRANSITIONS, f"Missing entry for {status}"

    def test_terminal_states_have_empty_transitions(self):
        for status in TERMINAL_STATES:
            assert VALID_TRANSITIONS[status] == set()
