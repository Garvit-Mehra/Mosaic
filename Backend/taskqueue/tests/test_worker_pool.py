"""Tests for the WorkerPool class.

Tests cover pool initialization, start/stop lifecycle, signal propagation,
and configurable worker count.

Requirements tested:
- Req 6.1: Configurable number of worker processes
- Req 6.6: Signal propagation for graceful shutdown
"""

import multiprocessing
import os
import signal
import sys
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.workers.pool import WorkerPool, GRACEFUL_SHUTDOWN_TIMEOUT


# --- Factory helpers for testing ---


def dummy_session_factory_creator():
    """Dummy factory that returns a mock session factory."""
    return MagicMock()


def dummy_redis_client_creator():
    """Dummy factory that returns a mock Redis client."""
    return MagicMock()


def dummy_priority_queue_creator():
    """Dummy factory that returns a mock priority queue."""
    return MagicMock()


def dummy_distributed_lock_creator():
    """Dummy factory that returns a mock distributed lock."""
    return MagicMock()


def dummy_handler_registry_creator():
    """Dummy factory that returns a mock handler registry."""
    return MagicMock()


@pytest.fixture
def pool_kwargs():
    """Default keyword arguments for creating a WorkerPool."""
    return {
        "worker_count": 3,
        "session_factory_creator": dummy_session_factory_creator,
        "redis_client_creator": dummy_redis_client_creator,
        "priority_queue_creator": dummy_priority_queue_creator,
        "distributed_lock_creator": dummy_distributed_lock_creator,
        "handler_registry_creator": dummy_handler_registry_creator,
    }


@pytest.fixture
def pool(pool_kwargs):
    """Create a WorkerPool instance (not started)."""
    return WorkerPool(**pool_kwargs)


# --- Test Classes ---


class TestWorkerPoolInit:
    """Test WorkerPool initialization."""

    def test_creates_with_worker_count(self, pool):
        """Pool should store the configured worker count (Req 6.1)."""
        assert pool.worker_count == 3

    def test_configurable_worker_count(self, pool_kwargs):
        """Worker count should be configurable (Req 6.1)."""
        pool_kwargs["worker_count"] = 8
        pool = WorkerPool(**pool_kwargs)
        assert pool.worker_count == 8

    def test_rejects_zero_worker_count(self, pool_kwargs):
        """Should reject worker_count of 0."""
        pool_kwargs["worker_count"] = 0
        with pytest.raises(ValueError, match="worker_count must be at least 1"):
            WorkerPool(**pool_kwargs)

    def test_rejects_negative_worker_count(self, pool_kwargs):
        """Should reject negative worker_count."""
        pool_kwargs["worker_count"] = -1
        with pytest.raises(ValueError, match="worker_count must be at least 1"):
            WorkerPool(**pool_kwargs)

    def test_starts_not_running(self, pool):
        """Pool should not be running initially."""
        assert pool.running is False

    def test_starts_with_no_processes(self, pool):
        """Pool should have no processes initially."""
        assert pool.processes == []
        assert pool.active_count == 0


class TestWorkerPoolStart:
    """Test WorkerPool.start() process spawning."""

    @patch("src.workers.pool.multiprocessing.Process")
    def test_spawns_configured_number_of_processes(self, mock_process_cls, pool):
        """start() should spawn worker_count processes (Req 6.1)."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        assert mock_process_cls.call_count == 3
        assert mock_process.start.call_count == 3

        # Clean up
        mock_process.is_alive.return_value = False
        pool._running = False

    @patch("src.workers.pool.multiprocessing.Process")
    def test_sets_running_flag(self, mock_process_cls, pool):
        """start() should set running to True."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        assert pool.running is True
        pool._running = False

    @patch("src.workers.pool.multiprocessing.Process")
    def test_raises_if_already_running(self, mock_process_cls, pool):
        """start() should raise RuntimeError if already running."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        with pytest.raises(RuntimeError, match="already running"):
            pool.start()

        pool._running = False

    @patch("src.workers.pool.multiprocessing.Process")
    def test_processes_are_not_daemon(self, mock_process_cls, pool):
        """Worker processes should not be daemons (so they can clean up)."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        # Verify daemon=False was passed
        for call_item in mock_process_cls.call_args_list:
            assert call_item[1]["daemon"] is False

        pool._running = False

    @patch("src.workers.pool.multiprocessing.Process")
    def test_processes_are_named(self, mock_process_cls, pool):
        """Worker processes should have descriptive names."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        names = [c[1]["name"] for c in mock_process_cls.call_args_list]
        assert names == ["worker-0", "worker-1", "worker-2"]

        pool._running = False


class TestWorkerPoolStop:
    """Test WorkerPool.stop() graceful and forceful shutdown."""

    @patch("src.workers.pool.multiprocessing.Process")
    @patch("src.workers.pool.os.kill")
    def test_graceful_stop_sends_sigterm(self, mock_kill, mock_process_cls, pool):
        """Graceful stop should send SIGTERM to all alive processes (Req 6.6)."""
        # Create separate mock processes so is_alive tracking is independent
        processes = []
        for i in range(3):
            p = MagicMock()
            p.pid = 1000 + i
            # Alive during SIGTERM send, then dead after join
            p.is_alive.side_effect = [True, False]
            processes.append(p)

        mock_process_cls.side_effect = processes

        pool.start()
        pool.stop(graceful=True)

        # SIGTERM should have been sent to each process
        assert mock_kill.call_count == 3
        for i in range(3):
            mock_kill.assert_any_call(1000 + i, signal.SIGTERM)

    @patch("src.workers.pool.multiprocessing.Process")
    def test_forceful_stop_terminates_processes(self, mock_process_cls, pool):
        """Forceful stop should call terminate() on all processes."""
        processes = []
        for i in range(3):
            p = MagicMock()
            p.pid = 1000 + i
            # Alive for terminate, then dead after join
            p.is_alive.side_effect = [True, False]
            processes.append(p)

        mock_process_cls.side_effect = processes

        pool.start()
        pool.stop(graceful=False)

        for p in processes:
            p.terminate.assert_called_once()

    @patch("src.workers.pool.multiprocessing.Process")
    def test_stop_clears_process_list(self, mock_process_cls, pool):
        """stop() should clear the internal process list."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.is_alive.return_value = False
        mock_process_cls.return_value = mock_process

        pool.start()
        pool.stop(graceful=True)

        assert pool.processes == []

    @patch("src.workers.pool.multiprocessing.Process")
    def test_stop_sets_running_false(self, mock_process_cls, pool):
        """stop() should set running to False."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.is_alive.return_value = False
        mock_process_cls.return_value = mock_process

        pool.start()
        pool.stop(graceful=True)

        assert pool.running is False

    def test_stop_when_not_running_is_noop(self, pool):
        """stop() when not running should be a no-op."""
        pool.stop(graceful=True)  # Should not raise
        assert pool.running is False

    @patch("src.workers.pool.multiprocessing.Process")
    @patch("src.workers.pool.os.kill")
    def test_graceful_stop_force_kills_stragglers(
        self, mock_kill, mock_process_cls, pool
    ):
        """Graceful stop should terminate processes that don't exit in time."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        # Process stays alive through SIGTERM and join
        mock_process.is_alive.return_value = True
        mock_process_cls.return_value = mock_process

        pool.start()

        # Simulate process not exiting: is_alive always True during stop
        # After terminate it finally dies
        alive_calls = [True] * 10 + [False] * 10
        mock_process.is_alive.side_effect = alive_calls

        pool.stop(graceful=True)

        # Should have been terminated after graceful timeout
        assert mock_process.terminate.call_count >= 1


class TestWorkerPoolSignalPropagation:
    """Test signal handler registration and propagation (Req 6.6)."""

    @patch("src.workers.pool.multiprocessing.Process")
    @patch("src.workers.pool.signal.signal")
    def test_registers_signal_handlers_on_start(
        self, mock_signal, mock_process_cls, pool
    ):
        """start() should register SIGTERM and SIGINT handlers."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()

        # Should have registered handlers for SIGTERM and SIGINT
        registered_signals = [c[0][0] for c in mock_signal.call_args_list]
        assert signal.SIGTERM in registered_signals
        if sys.platform != "win32":
            assert signal.SIGINT in registered_signals

        pool._running = False

    @patch("src.workers.pool.multiprocessing.Process")
    def test_signal_handler_triggers_graceful_stop(self, mock_process_cls, pool):
        """Signal handler should trigger graceful stop (Req 6.6)."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.is_alive.return_value = False
        mock_process_cls.return_value = mock_process

        pool.start()

        # Simulate receiving SIGTERM by calling the signal handler directly
        pool._signal_handler(signal.SIGTERM, None)

        # Pool should have stopped
        assert pool.running is False

    @patch("src.workers.pool.multiprocessing.Process")
    @patch("src.workers.pool.signal.signal")
    @patch("src.workers.pool.signal.getsignal")
    def test_restores_original_signal_handlers_on_stop(
        self, mock_getsignal, mock_signal_fn, mock_process_cls, pool
    ):
        """stop() should restore original signal handlers."""
        original_handler = MagicMock()
        mock_getsignal.return_value = original_handler

        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.is_alive.return_value = False
        mock_process_cls.return_value = mock_process

        pool.start()
        pool.stop(graceful=True)

        # Signal handlers should have been restored
        # The last calls to signal.signal should restore the originals
        restore_calls = [
            c for c in mock_signal_fn.call_args_list
            if c[0][1] == original_handler
        ]
        assert len(restore_calls) >= 1


class TestWorkerPoolActiveCount:
    """Test active_count property."""

    @patch("src.workers.pool.multiprocessing.Process")
    def test_active_count_reflects_alive_processes(self, mock_process_cls, pool):
        """active_count should count alive processes."""
        processes = []
        for i in range(3):
            p = MagicMock()
            p.pid = 1000 + i
            p.is_alive.return_value = (i < 2)  # 2 alive, 1 dead
            processes.append(p)

        mock_process_cls.side_effect = processes

        pool.start()

        assert pool.active_count == 2

        pool._running = False


class TestWorkerPoolWait:
    """Test wait() blocking behavior."""

    @patch("src.workers.pool.multiprocessing.Process")
    def test_wait_joins_all_processes(self, mock_process_cls, pool):
        """wait() should join all processes."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process_cls.return_value = mock_process

        pool.start()
        pool.wait()

        # Each process should have been joined
        assert mock_process.join.call_count == 3
