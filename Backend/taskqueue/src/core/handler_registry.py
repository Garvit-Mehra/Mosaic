"""Handler registry for job type to handler function mapping.

Allows registering callable handlers by job_type string and looking them up
at validation time and execution time.
"""

from typing import Callable, Dict, List, Optional


class HandlerRegistry:
    """Registry that maps job_type strings to handler callables.

    Thread-safe for reads after initial registration. Handlers should be
    registered at application startup before workers begin processing.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable] = {}

    def register(self, job_type: str, handler: Callable) -> None:
        """Register a handler function for a given job type.

        Args:
            job_type: The string identifier for the job type.
            handler: A callable that will process jobs of this type.

        Raises:
            ValueError: If job_type is empty or handler is not callable.
        """
        if not job_type:
            raise ValueError("job_type must be a non-empty string")
        if not callable(handler):
            raise ValueError("handler must be callable")
        self._handlers[job_type] = handler

    def get(self, job_type: str) -> Optional[Callable]:
        """Look up the handler for a given job type.

        Args:
            job_type: The job type to look up.

        Returns:
            The registered handler callable, or None if not registered.
        """
        return self._handlers.get(job_type)

    def is_registered(self, job_type: str) -> bool:
        """Check whether a job type has a registered handler.

        Args:
            job_type: The job type to check.

        Returns:
            True if the job type is registered, False otherwise.
        """
        return job_type in self._handlers

    def list_types(self) -> List[str]:
        """List all registered job types.

        Returns:
            A sorted list of registered job type strings.
        """
        return sorted(self._handlers.keys())

    def unregister(self, job_type: str) -> bool:
        """Remove a handler registration.

        Args:
            job_type: The job type to unregister.

        Returns:
            True if the handler was found and removed, False if not registered.
        """
        if job_type in self._handlers:
            del self._handlers[job_type]
            return True
        return False


# Global handler registry instance
handler_registry = HandlerRegistry()
