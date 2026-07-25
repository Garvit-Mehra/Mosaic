"""Property-based tests for exponential backoff monotonicity and bounds.

**Validates: Requirements 9.2, 9.3, 9.4**

Uses Hypothesis to generate retry counts and base values to verify:
- Property 5: Backoff delays are monotonically non-decreasing for sequential
  retry counts when base >= 1.0 (Req 9.4)
- Property 6: Backoff delays are always positive and never exceed the maximum
  allowed bound of max_delay + 0.1 * max_delay (Req 9.2, 9.3)
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.backoff import calculate_backoff


# --- Strategies ---

# Retry counts: reasonable range for property testing
retry_count_strategy = st.integers(min_value=0, max_value=50)

# Base values >= 1.0 for monotonicity guarantee (Req 9.4)
monotonic_base_strategy = st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Base values > 0 for general bounds checking
positive_base_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

# Max delay values: reasonable positive range
max_delay_strategy = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


# --- Property Tests ---


class TestBackoffMonotonicity:
    """Property 5: Backoff Monotonicity.

    **Validates: Requirements 9.4**

    For any base >= 1.0 and sequential retry counts (n, n+1),
    delay(n) <= delay(n+1) when jitter is zero.
    """

    @given(
        retry_count=st.integers(min_value=0, max_value=49),
        base=monotonic_base_strategy,
        max_delay=max_delay_strategy,
    )
    @settings(max_examples=500)
    def test_sequential_delays_non_decreasing_without_jitter(self, retry_count, base, max_delay):
        """delay(n) <= delay(n+1) for base >= 1.0 with zero jitter.

        **Validates: Requirements 9.4**
        """
        delay_n = calculate_backoff(
            retry_count=retry_count,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: 0.0,
        )
        delay_n_plus_1 = calculate_backoff(
            retry_count=retry_count + 1,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: 0.0,
        )

        assert delay_n <= delay_n_plus_1, (
            f"Monotonicity violated: delay({retry_count})={delay_n} > "
            f"delay({retry_count + 1})={delay_n_plus_1} "
            f"with base={base}, max_delay={max_delay}"
        )

    @given(
        base=monotonic_base_strategy,
        max_delay=max_delay_strategy,
    )
    @settings(max_examples=200)
    def test_full_sequence_non_decreasing(self, base, max_delay):
        """An entire sequence of delays (0..20) is non-decreasing with zero jitter.

        **Validates: Requirements 9.4**
        """
        delays = [
            calculate_backoff(
                retry_count=i,
                base=base,
                max_delay=max_delay,
                random_func=lambda lo, hi: 0.0,
            )
            for i in range(21)
        ]

        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1], (
                f"Monotonicity violated at step {i}: "
                f"delay({i - 1})={delays[i - 1]} > delay({i})={delays[i]} "
                f"with base={base}, max_delay={max_delay}"
            )


class TestBackoffBounds:
    """Property 6: Backoff Bounds.

    **Validates: Requirements 9.2, 9.3**

    - Delay is always > 0 for positive base (Req 9.2)
    - Delay without jitter never exceeds max_delay (Req 9.3)
    - Delay with maximum jitter never exceeds max_delay * 1.1 (Req 9.3)
    """

    @given(
        retry_count=retry_count_strategy,
        base=positive_base_strategy,
        max_delay=max_delay_strategy,
    )
    @settings(max_examples=500)
    def test_delay_always_positive(self, retry_count, base, max_delay):
        """Delay is always > 0 for any positive base and valid retry count.

        **Validates: Requirements 9.2**
        """
        # Test with zero jitter
        delay = calculate_backoff(
            retry_count=retry_count,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: 0.0,
        )
        assert delay > 0, (
            f"Delay must be positive, got {delay} "
            f"with retry_count={retry_count}, base={base}, max_delay={max_delay}"
        )

    @given(
        retry_count=retry_count_strategy,
        base=positive_base_strategy,
        max_delay=max_delay_strategy,
    )
    @settings(max_examples=500)
    def test_delay_without_jitter_capped_at_max_delay(self, retry_count, base, max_delay):
        """Delay without jitter never exceeds max_delay.

        **Validates: Requirements 9.3**
        """
        delay = calculate_backoff(
            retry_count=retry_count,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: 0.0,
        )
        assert delay <= max_delay, (
            f"Delay without jitter ({delay}) exceeds max_delay ({max_delay}) "
            f"with retry_count={retry_count}, base={base}"
        )

    @given(
        retry_count=retry_count_strategy,
        base=positive_base_strategy,
        max_delay=max_delay_strategy,
    )
    @settings(max_examples=500)
    def test_delay_with_max_jitter_bounded(self, retry_count, base, max_delay):
        """Delay with maximum jitter never exceeds max_delay * 1.1.

        When jitter is at its maximum (0.1 * capped_delay), the total delay
        is capped_delay + 0.1 * capped_delay = 1.1 * capped_delay <= 1.1 * max_delay.

        **Validates: Requirements 9.3**
        """
        # Use max jitter: random_func returns the high bound
        delay = calculate_backoff(
            retry_count=retry_count,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: hi,
        )
        upper_bound = max_delay * 1.1
        assert delay <= upper_bound + 1e-9, (
            f"Delay with max jitter ({delay}) exceeds 1.1 * max_delay ({upper_bound}) "
            f"with retry_count={retry_count}, base={base}, max_delay={max_delay}"
        )

    @given(
        retry_count=retry_count_strategy,
        base=positive_base_strategy,
        max_delay=max_delay_strategy,
        jitter_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=500)
    def test_delay_with_any_jitter_bounded(self, retry_count, base, max_delay, jitter_fraction):
        """Delay with any valid jitter value is bounded by 1.1 * max_delay.

        **Validates: Requirements 9.2, 9.3**
        """
        # Simulate any jitter within its valid range [0, 0.1 * capped_delay]
        delay = calculate_backoff(
            retry_count=retry_count,
            base=base,
            max_delay=max_delay,
            random_func=lambda lo, hi: lo + jitter_fraction * (hi - lo),
        )
        upper_bound = max_delay * 1.1
        assert delay <= upper_bound + 1e-9, (
            f"Delay ({delay}) exceeds upper bound ({upper_bound}) "
            f"with retry_count={retry_count}, base={base}, "
            f"max_delay={max_delay}, jitter_fraction={jitter_fraction}"
        )
        assert delay > 0, (
            f"Delay must be positive, got {delay}"
        )
