"""Worker pool management using Python multiprocessing.

Spawns a configurable number of worker processes, each running its own
asyncio event loop. Supports graceful and forceful shutdown with signal
propagation to child processes.

Requirements covered:
- Req 6.1: Configurable number of worker processes
- Req 6.6: Signal propagation for graceful shutdown
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Callable, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Default timeout for graceful shutdown (seconds)
GRACEFUL_SHUTDOWN_TIMEOUT = 30


def _worker_process_target(
    session_factory_creator: Callable,
    redis_client_creator: Callable,
    priority_queue_creator: Callable,
    distributed_lock_creator: Callable,
    handler_registry_creator: Callable,
) -> None:
    """Target function for each worker process.

    Creates a fresh asyncio event loop, instantiates all dependencies
    from factory functions (since connections cannot be shared across
    process boundaries), creates a Worker instance, and runs it.

    Args:
        session_factory_creator: Callable that returns an async session factory.
        redis_client_creator: Callable that returns an async Redis client.
        priority_queue_creator: Callable that returns a PriorityQueueInterface.
        distributed_lock_creator: Callable that returns a DistributedLock.
        handler_registry_creator: Callable that returns a HandlerRegistry.
    """
    from src.workers.worker import Worker

    # Each process gets its own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create fresh connections per process (connections are not fork-safe)
    session_factory = session_factory_creator()
    redis_client = redis_client_creator()
    priority_queue = priority_queue_creator()
    distributed_lock = distributed_lock_creator()
    handler_registry = handler_registry_creator()

    # Create worker with unique ID
    worker_id = uuid4()
    worker = Worker(
        worker_id=worker_id,
        session_factory=session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
        distributed_lock=distributed_lock,
        handler_registry=handler_registry,
    )

    logger.info(f"Worker process started (pid={os.getpid()}, worker_id={worker_id})")

    try:
        loop.run_until_complete(worker.start())
    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} (pid={os.getpid()}) received KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Worker {worker_id} (pid={os.getpid()}) crashed: {e}")
    finally:
        # Clean up the event loop
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        logger.info(f"Worker process exited (pid={os.getpid()}, worker_id={worker_id})")


class WorkerPool:
    """Manages a pool of worker processes using Python multiprocessing.

    Each worker process runs its own asyncio event loop and Worker instance.
    The pool handles starting, stopping, and signal propagation to all
    child processes.

    Requirements:
    - Req 6.1: Configurable number of worker processes
    - Req 6.6: Signal propagation for graceful shutdown
    """

    def __init__(
        self,
        worker_count: int,
        session_factory_creator: Callable,
        redis_client_creator: Callable,
        priority_queue_creator: Callable,
        distributed_lock_creator: Callable,
        handler_registry_creator: Callable,
    ) -> None:
        """Initialize the worker pool.

        Args:
            worker_count: Number of worker processes to spawn.
            session_factory_creator: Factory function that creates a new session factory.
            redis_client_creator: Factory function that creates a new Redis client.
            priority_queue_creator: Factory function that creates a new PriorityQueue.
            distributed_lock_creator: Factory function that creates a new DistributedLock.
            handler_registry_creator: Factory function that creates a HandlerRegistry.
        """
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")

        self._worker_count = worker_count
        self._session_factory_creator = session_factory_creator
        self._redis_client_creator = redis_client_creator
        self._priority_queue_creator = priority_queue_creator
        self._distributed_lock_creator = distributed_lock_creator
        self._handler_registry_creator = handler_registry_creator

        self._processes: List[multiprocessing.Process] = []
        self._running = False
        self._original_sigterm_handler = None
        self._original_sigint_handler = None

    @property
    def worker_count(self) -> int:
        """Number of worker processes configured."""
        return self._worker_count

    @property
    def running(self) -> bool:
        """Whether the pool is currently running."""
        return self._running

    @property
    def processes(self) -> List[multiprocessing.Process]:
        """List of managed worker processes."""
        return list(self._processes)

    @property
    def active_count(self) -> int:
        """Number of currently alive worker processes."""
        return sum(1 for p in self._processes if p.is_alive())

    def start(self) -> None:
        """Start all worker processes.

        Spawns worker_count processes, each running _worker_process_target.
        Registers signal handlers on the pool manager process to propagate
        signals to all children.

        Raises:
            RuntimeError: If the pool is already running.
        """
        if self._running:
            raise RuntimeError("Worker pool is already running")

        logger.info(f"Starting worker pool with {self._worker_count} processes")

        # Register signal handlers for signal propagation (Req 6.6)
        self._register_signal_handlers()

        # Spawn worker processes
        for i in range(self._worker_count):
            process = multiprocessing.Process(
                target=_worker_process_target,
                args=(
                    self._session_factory_creator,
                    self._redis_client_creator,
                    self._priority_queue_creator,
                    self._distributed_lock_creator,
                    self._handler_registry_creator,
                ),
                name=f"worker-{i}",
                daemon=False,
            )
            process.start()
            self._processes.append(process)
            logger.info(f"Started worker process {i} (pid={process.pid})")

        self._running = True
        logger.info(f"Worker pool started: {self._worker_count} processes running")

    def stop(self, graceful: bool = True) -> None:
        """Stop all worker processes.

        Args:
            graceful: If True, send SIGTERM and wait for processes to finish
                     (up to GRACEFUL_SHUTDOWN_TIMEOUT). If False, send SIGKILL
                     / terminate immediately.
        """
        if not self._running:
            return

        logger.info(
            f"Stopping worker pool ({'graceful' if graceful else 'forceful'})"
        )

        if graceful:
            self._graceful_stop()
        else:
            self._forceful_stop()

        # Restore original signal handlers
        self._restore_signal_handlers()

        self._processes.clear()
        self._running = False
        logger.info("Worker pool stopped")

    def _graceful_stop(self) -> None:
        """Gracefully stop all processes: SIGTERM → wait → SIGKILL stragglers."""
        # Send SIGTERM to all alive processes
        for process in self._processes:
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to worker (pid={process.pid})")
                except (ProcessLookupError, OSError):
                    # Process already exited
                    pass

        # Wait for processes to finish with timeout
        deadline = time.time() + GRACEFUL_SHUTDOWN_TIMEOUT
        for process in self._processes:
            remaining = max(0, deadline - time.time())
            process.join(timeout=remaining)

        # Force-kill any processes that didn't exit in time
        for process in self._processes:
            if process.is_alive():
                logger.warning(
                    f"Worker (pid={process.pid}) did not exit gracefully, terminating"
                )
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)

    def _forceful_stop(self) -> None:
        """Forcefully stop all processes immediately."""
        for process in self._processes:
            if process.is_alive():
                try:
                    process.terminate()
                    logger.debug(f"Terminated worker (pid={process.pid})")
                except (ProcessLookupError, OSError):
                    pass

        # Brief wait for termination
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    def _register_signal_handlers(self) -> None:
        """Register signal handlers to propagate signals to child processes.

        When the pool manager receives SIGTERM or SIGINT, it propagates
        the signal to all child worker processes for graceful shutdown (Req 6.6).
        """
        if sys.platform == "win32":
            # Windows doesn't support SIGTERM via signal module in the same way
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            self._original_sigterm_handler = None

        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGTERM/SIGINT by initiating graceful pool shutdown.

        Propagates the signal to all child processes (Req 6.6).
        """
        sig_name = signal.Signals(signum).name
        logger.info(f"Worker pool received {sig_name}, initiating graceful shutdown")
        self.stop(graceful=True)

    def wait(self) -> None:
        """Block until all worker processes have exited.

        Useful for keeping the main process alive while workers run.
        """
        for process in self._processes:
            process.join()
        self._running = False
