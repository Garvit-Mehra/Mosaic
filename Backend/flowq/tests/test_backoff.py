"""Unit tests for exponential backoff calculation."""

import pytest

from src.core.backoff import calculate_backoff


class TestCalculateBackoff:
    """Tests for calculate_backoff function."""

    def test_basic_calculation_no_jitter(self):
        """With zero jitter, delay equals base^retry_count."""
        # Use a deterministic random_func that always returns 0 (no jitter)
        result = calculate_backoff(retry_count=1, base=2.0, random_func=lambda lo, hi: 0.0)
        assert result == 2.0  # 2^1 = 2, jitter = 0

    def test_retry_count_zero(self):
        """retry_count=0 gives base^0 = 1.0 as base delay."""
        result = calculate_backoff(retry_count=0, base=2.0, random_func=lambda lo, hi: 0.0)
        assert result == 1.0  # 2^0 = 1

    def test_exponential_growth(self):
        """Delays grow exponentially with retry_count."""
        delays = [
            calculate_backoff(retry_count=i, base=2.0, random_func=lambda lo, hi: 0.0)
            for i in range(5)
        ]
        # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        result = calculate_backoff(
            retry_count=10, base=2.0, max_delay=300.0, random_func=lambda lo, hi: 0.0
        )
        # 2^10 = 1024, but capped at 300
        assert result == 300.0

    def test_jitter_added_on_top(self):
        """Jitter is added on top of the capped delay."""
        # Use max jitter (0.1 * capped_delay)
        result = calculate_backoff(
            retry_count=2, base=2.0, random_func=lambda lo, hi: hi
        )
        # 2^2 = 4, jitter = 0.1 * 4 = 0.4, total = 4.4
        assert result == pytest.approx(4.4)

    def test_jitter_range(self):
        """Jitter random_func is called with correct range [0, 0.1 * capped_delay]."""
        captured_args = []

        def capture_random(lo, hi):
            captured_args.append((lo, hi))
            return 0.0

        calculate_backoff(retry_count=3, base=2.0, random_func=capture_random)
        # 2^3 = 8, jitter range should be [0, 0.8]
        assert captured_args[0] == (0.0, pytest.approx(0.8))

    def test_jitter_when_capped(self):
        """When delay is capped, jitter is based on max_delay."""
        captured_args = []

        def capture_random(lo, hi):
            captured_args.append((lo, hi))
            return 0.0

        calculate_backoff(
            retry_count=20, base=2.0, max_delay=300.0, random_func=capture_random
        )
        # 2^20 = 1048576, capped at 300, jitter range = [0, 0.1 * 300] = [0, 30]
        assert captured_args[0] == (0.0, pytest.approx(30.0))

    def test_monotonically_non_decreasing_without_jitter(self):
        """Base delays are monotonically non-decreasing."""
        delays = [
            calculate_backoff(retry_count=i, base=2.0, random_func=lambda lo, hi: 0.0)
            for i in range(20)
        ]
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]

    def test_base_one_constant_delay(self):
        """With base=1.0, all delays are 1.0 (no growth)."""
        delays = [
            calculate_backoff(retry_count=i, base=1.0, random_func=lambda lo, hi: 0.0)
            for i in range(5)
        ]
        assert all(d == 1.0 for d in delays)

    def test_custom_base(self):
        """Custom base value works correctly."""
        result = calculate_backoff(retry_count=3, base=3.0, random_func=lambda lo, hi: 0.0)
        assert result == 27.0  # 3^3 = 27

    def test_custom_max_delay(self):
        """Custom max_delay value caps correctly."""
        result = calculate_backoff(
            retry_count=5, base=2.0, max_delay=20.0, random_func=lambda lo, hi: 0.0
        )
        # 2^5 = 32, capped at 20
        assert result == 20.0

    def test_delay_always_positive(self):
        """Delay is always positive for valid inputs."""
        result = calculate_backoff(retry_count=0, base=0.5, random_func=lambda lo, hi: 0.0)
        assert result > 0

    def test_invalid_negative_retry_count(self):
        """Raises ValueError for negative retry_count."""
        with pytest.raises(ValueError, match="retry_count must be non-negative"):
            calculate_backoff(retry_count=-1)

    def test_invalid_zero_base(self):
        """Raises ValueError for zero base."""
        with pytest.raises(ValueError, match="base must be positive"):
            calculate_backoff(retry_count=1, base=0.0)

    def test_invalid_negative_base(self):
        """Raises ValueError for negative base."""
        with pytest.raises(ValueError, match="base must be positive"):
            calculate_backoff(retry_count=1, base=-1.0)

    def test_invalid_zero_max_delay(self):
        """Raises ValueError for zero max_delay."""
        with pytest.raises(ValueError, match="max_delay must be positive"):
            calculate_backoff(retry_count=1, max_delay=0.0)

    def test_invalid_negative_max_delay(self):
        """Raises ValueError for negative max_delay."""
        with pytest.raises(ValueError, match="max_delay must be positive"):
            calculate_backoff(retry_count=1, max_delay=-10.0)

    def test_default_parameters(self):
        """Function works with default parameters (uses random jitter)."""
        result = calculate_backoff(retry_count=1)
        # base=2, retry_count=1 -> 2^1=2, jitter in [0, 0.2]
        # so result should be in [2.0, 2.2]
        assert 2.0 <= result <= 2.2

    def test_large_retry_count_capped(self):
        """Very large retry counts are handled gracefully due to cap."""
        result = calculate_backoff(
            retry_count=100, base=2.0, max_delay=300.0, random_func=lambda lo, hi: 0.0
        )
        assert result == 300.0
