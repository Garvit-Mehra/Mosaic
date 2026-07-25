"""Unit tests for the handler registry."""

import pytest

from src.core.handler_registry import HandlerRegistry


def sample_handler(payload: dict) -> dict:
    """A sample handler for testing."""
    return {"processed": True}


async def async_handler(payload: dict) -> dict:
    """An async sample handler for testing."""
    return {"async": True}


class TestHandlerRegistry:
    """Tests for HandlerRegistry class."""

    def test_register_and_get(self):
        registry = HandlerRegistry()
        registry.register("email", sample_handler)
        assert registry.get("email") is sample_handler

    def test_get_unregistered_returns_none(self):
        registry = HandlerRegistry()
        assert registry.get("nonexistent") is None

    def test_is_registered_true(self):
        registry = HandlerRegistry()
        registry.register("email", sample_handler)
        assert registry.is_registered("email") is True

    def test_is_registered_false(self):
        registry = HandlerRegistry()
        assert registry.is_registered("email") is False

    def test_list_types_empty(self):
        registry = HandlerRegistry()
        assert registry.list_types() == []

    def test_list_types_sorted(self):
        registry = HandlerRegistry()
        registry.register("webhook", sample_handler)
        registry.register("email", sample_handler)
        registry.register("analytics", sample_handler)
        assert registry.list_types() == ["analytics", "email", "webhook"]

    def test_register_overwrites_existing(self):
        registry = HandlerRegistry()
        registry.register("email", sample_handler)
        registry.register("email", async_handler)
        assert registry.get("email") is async_handler

    def test_register_empty_job_type_raises(self):
        registry = HandlerRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register("", sample_handler)

    def test_register_non_callable_raises(self):
        registry = HandlerRegistry()
        with pytest.raises(ValueError, match="callable"):
            registry.register("email", "not_a_callable")  # type: ignore

    def test_unregister_existing(self):
        registry = HandlerRegistry()
        registry.register("email", sample_handler)
        assert registry.unregister("email") is True
        assert registry.is_registered("email") is False

    def test_unregister_nonexistent(self):
        registry = HandlerRegistry()
        assert registry.unregister("nonexistent") is False

    def test_register_async_handler(self):
        registry = HandlerRegistry()
        registry.register("async_job", async_handler)
        assert registry.get("async_job") is async_handler
        assert registry.is_registered("async_job") is True

    def test_register_lambda_handler(self):
        registry = HandlerRegistry()
        handler = lambda payload: payload  # noqa: E731
        registry.register("lambda_job", handler)
        assert registry.get("lambda_job") is handler

    def test_register_class_instance_handler(self):
        class MyHandler:
            def __call__(self, payload: dict) -> dict:
                return payload

        registry = HandlerRegistry()
        handler = MyHandler()
        registry.register("class_job", handler)
        assert registry.get("class_job") is handler
