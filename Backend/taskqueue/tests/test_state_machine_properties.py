"""Property-based tests for job state machine validity.

**Validates: Requirements 12.1, 12.2, 12.4**

Uses Hypothesis to generate random state transitions and verify:
- Only valid transitions succeed (Req 12.1)
- Invalid transitions are rejected without modifying job state (Req 12.2)
- Terminal states reject ALL outgoing transitions (Req 12.4)
"""

from dataclasses import dataclass, field
from datetime import datetime

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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


# --- Strategies ---

job_status_strategy = st.sampled_from(list(JobStatus))
terminal_status_strategy = st.sampled_from(list(TERMINAL_STATES))
non_terminal_status_strategy = st.sampled_from(
    [s for s in JobStatus if s not in TERMINAL_STATES]
)


def transition_sequence_strategy(min_size=1, max_size=10):
    """Generate a random sequence of target states to attempt transitioning through."""
    return st.lists(job_status_strategy, min_size=min_size, max_size=max_size)


# --- Property Tests ---


class TestStateMachineValidity:
    """Property 3: State Machine Validity.

    **Validates: Requirements 12.1, 12.2, 12.4**
    """

    @given(current=job_status_strategy, target=job_status_strategy)
    @settings(max_examples=200)
    def test_valid_transitions_match_transitions_map(self, current, target):
        """validate_transition returns True iff target is in VALID_TRANSITIONS[current].

        **Validates: Requirements 12.1**
        """
        expected = target in VALID_TRANSITIONS.get(current, set())
        assert validate_transition(current, target) == expected

    @given(current=job_status_strategy, target=job_status_strategy)
    @settings(max_examples=200)
    def test_invalid_transitions_leave_job_unchanged(self, current, target):
        """Invalid transitions must not modify the job's status or updated_at.

        **Validates: Requirements 12.2**
        """
        assume(not validate_transition(current, target))

        job = FakeJob(status=current)
        original_status = job.status
        original_updated_at = job.updated_at

        with pytest.raises(InvalidTransitionError):
            apply_transition(job, target)

        # Job must be completely unchanged
        assert job.status == original_status
        assert job.updated_at == original_updated_at

    @given(terminal=terminal_status_strategy, target=job_status_strategy)
    @settings(max_examples=200)
    def test_terminal_states_reject_all_outgoing_transitions(self, terminal, target):
        """Terminal states (COMPLETED, CANCELLED, DEAD_LETTER) must reject ALL transitions.

        **Validates: Requirements 12.4**
        """
        # validate_transition should return False for any target from a terminal state
        assert validate_transition(terminal, target) is False

        # apply_transition should raise and leave job unchanged
        job = FakeJob(status=terminal)
        original_updated_at = job.updated_at

        with pytest.raises(InvalidTransitionError) as exc_info:
            apply_transition(job, target)

        assert job.status == terminal
        assert job.updated_at == original_updated_at
        assert "terminal state" in str(exc_info.value)

    @given(
        initial=non_terminal_status_strategy,
        targets=transition_sequence_strategy(min_size=1, max_size=10),
    )
    @settings(max_examples=300)
    def test_random_transition_sequences_maintain_invariants(self, initial, targets):
        """Random sequences of transitions maintain state machine invariants.

        For each attempted transition in the sequence:
        - If valid: job status updates to target
        - If invalid: job status remains at its current value

        **Validates: Requirements 12.1, 12.2**
        """
        job = FakeJob(status=initial)

        for target in targets:
            current_before = job.status
            updated_before = job.updated_at

            if validate_transition(current_before, target):
                # Valid transition: should succeed and update state
                apply_transition(job, target)
                assert job.status == target
                assert job.updated_at >= updated_before
            else:
                # Invalid transition: should fail and leave state unchanged
                with pytest.raises(InvalidTransitionError):
                    apply_transition(job, target)
                assert job.status == current_before
                assert job.updated_at == updated_before

    @given(data=st.data())
    @settings(max_examples=200)
    def test_valid_transitions_update_both_status_and_timestamp(self, data):
        """Valid transitions atomically update both status and updated_at.

        **Validates: Requirements 12.1**
        """
        # Only pick from states that have at least one valid outgoing transition
        valid_sources = [s for s in JobStatus if VALID_TRANSITIONS.get(s)]
        current = data.draw(st.sampled_from(valid_sources))
        target = data.draw(st.sampled_from(list(VALID_TRANSITIONS[current])))

        job = FakeJob(status=current)
        old_updated = job.updated_at

        apply_transition(job, target)

        assert job.status == target
        assert job.updated_at >= old_updated
