"""Integration tests for plugin registry FastAPI lifespan wiring."""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

from cli_agent_orchestrator.api.main import app, get_plugin_registry, lifespan
from cli_agent_orchestrator.plugins import CaoPlugin, PluginRegistry, hook
from cli_agent_orchestrator.plugins.events import PostSendMessageEvent


async def fake_flow_daemon() -> None:
    """Minimal async flow daemon stub for lifespan tests."""


async def fake_opencode_daemon(registry) -> None:
    """Minimal async OpenCode inbox poller stub for lifespan tests."""
    del registry


def _consumer_patches():
    """Patch the event-bus consumer coroutines and the OpenCode poller.

    The merged lifespan starts ``status_monitor.run()``, ``log_writer.run()``,
    ``inbox_service.run()`` and ``assigned_worker_completion_service.run()`` as
    background tasks (each an endless event-bus consumer loop) and the
    OpenCode inbox delivery daemon. These must be stubbed so the lifespan
    enters and exits cleanly without spinning real consumer loops or leaving
    un-awaited coroutines. ``run`` is an async method on each singleton, so an
    ``AsyncMock`` yields an awaitable that ``asyncio.create_task`` can
    schedule and complete immediately.

    ``register_persisted_assignments()`` is also stubbed here even though it
    is a synchronous *startup* call, not a background consumer: it queries
    the real, shared ``assigned_worker_callbacks`` table exactly like the
    consumer loops above, so these lifespan-wiring tests must not depend on
    that table's on-disk schema being current for whatever this test process
    happens to be pointed at.
    """
    return (
        patch("cli_agent_orchestrator.api.main.status_monitor.run", new_callable=AsyncMock),
        patch("cli_agent_orchestrator.api.main.log_writer.run", new_callable=AsyncMock),
        patch("cli_agent_orchestrator.api.main.inbox_service.run", new_callable=AsyncMock),
        patch(
            "cli_agent_orchestrator.api.main.assigned_worker_completion_service.run",
            new_callable=AsyncMock,
        ),
        patch(
            "cli_agent_orchestrator.api.main.assigned_worker_completion_service.register_persisted_assignments"
        ),
        patch(
            "cli_agent_orchestrator.api.main.opencode_inbox_delivery_daemon",
            fake_opencode_daemon,
        ),
    )


class TestPluginRegistryLifespan:
    """Tests for plugin registry startup, app state wiring, and teardown."""

    @pytest.mark.asyncio
    async def test_lifespan_stores_registry_and_tears_it_down(self) -> None:
        """The lifespan should create, store, expose, and tear down the registry."""

        ordering: list[str] = []
        mock_load = AsyncMock()
        mock_teardown = AsyncMock()
        mock_load.side_effect = lambda: ordering.append("registry_load")

        request_scope = {"type": "http", "app": app, "headers": []}

        (
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
        ) = _consumer_patches()

        with (
            patch("cli_agent_orchestrator.api.main.setup_logging"),
            patch("cli_agent_orchestrator.api.main.init_db"),
            patch(
                "cli_agent_orchestrator.services.memory_reconciliation.reconcile_memory_startup",
                return_value=None,
            ),
            patch("cli_agent_orchestrator.api.main.cleanup_old_data"),
            patch(
                "cli_agent_orchestrator.api.main.cleanup_expired_memories", new_callable=AsyncMock
            ),
            patch("cli_agent_orchestrator.api.main.flow_daemon", fake_flow_daemon),
            patch("cli_agent_orchestrator.api.main.bus.set_loop"),
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
            patch.object(PluginRegistry, "load", mock_load),
            patch.object(PluginRegistry, "teardown", mock_teardown),
        ):
            async with lifespan(app):
                registry = app.state.plugin_registry

                assert isinstance(registry, PluginRegistry)
                assert get_plugin_registry(Request(request_scope)) is registry
                assert get_plugin_registry(Request(dict(request_scope))) is registry
                mock_load.assert_awaited_once()
                # Registry load is the first startup step.
                assert ordering == ["registry_load"]

            mock_teardown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_logs_no_plugins_registered_when_entry_points_are_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The lifespan should surface the empty-plugin INFO log from the registry."""

        (
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
        ) = _consumer_patches()

        with (
            patch("cli_agent_orchestrator.api.main.setup_logging"),
            patch("cli_agent_orchestrator.api.main.init_db"),
            patch(
                "cli_agent_orchestrator.services.memory_reconciliation.reconcile_memory_startup",
                return_value=None,
            ),
            patch("cli_agent_orchestrator.api.main.cleanup_old_data"),
            patch(
                "cli_agent_orchestrator.api.main.cleanup_expired_memories", new_callable=AsyncMock
            ),
            patch("cli_agent_orchestrator.api.main.flow_daemon", fake_flow_daemon),
            patch("cli_agent_orchestrator.api.main.bus.set_loop"),
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
            patch("importlib.metadata.entry_points", return_value=[]),
        ):
            with caplog.at_level(logging.INFO, logger="cli_agent_orchestrator.plugins.registry"):
                async with lifespan(app):
                    assert isinstance(app.state.plugin_registry, PluginRegistry)

        assert "No CAO plugins registered" in caplog.text

    @pytest.mark.asyncio
    async def test_lifespan_tolerates_plugin_setup_failure(self) -> None:
        """The lifespan should still start when one plugin fails during setup."""

        class FailingPlugin(CaoPlugin):
            async def setup(self) -> None:
                raise RuntimeError("setup failed")

        class HealthyPlugin(CaoPlugin):
            @hook("post_send_message")
            async def on_message(self, event: PostSendMessageEvent) -> None:
                del event

        (
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
        ) = _consumer_patches()

        with (
            patch("cli_agent_orchestrator.api.main.setup_logging"),
            patch("cli_agent_orchestrator.api.main.init_db"),
            patch(
                "cli_agent_orchestrator.services.memory_reconciliation.reconcile_memory_startup",
                return_value=None,
            ),
            patch("cli_agent_orchestrator.api.main.cleanup_old_data"),
            patch(
                "cli_agent_orchestrator.api.main.cleanup_expired_memories", new_callable=AsyncMock
            ),
            patch("cli_agent_orchestrator.api.main.flow_daemon", fake_flow_daemon),
            patch("cli_agent_orchestrator.api.main.bus.set_loop"),
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
            patch(
                "importlib.metadata.entry_points",
                return_value=[
                    type("EP", (), {"name": "failing", "load": lambda self: FailingPlugin})(),
                    type("EP", (), {"name": "healthy", "load": lambda self: HealthyPlugin})(),
                ],
            ),
        ):
            async with lifespan(app):
                registry = app.state.plugin_registry

                assert isinstance(registry, PluginRegistry)
                assert len(registry._plugins) == 1


class TestLifespanShutdownDrainsNativeInputLocks:
    """Correction-1038 Part C.3: shutdown must prove no physical native-
    input write is in flight -- for any known terminal -- before this
    process is considered safe to replace. Pins that the real ``lifespan``
    shutdown sequence actually calls ``drain_all_terminal_input_locks``
    (not just that the primitive itself works in isolation, which
    ``test_native_input_serialization_race.py`` already covers).
    """

    @pytest.mark.asyncio
    async def test_shutdown_calls_drain_all_terminal_input_locks(self) -> None:
        (
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
        ) = _consumer_patches()
        mock_teardown = AsyncMock()

        with (
            patch("cli_agent_orchestrator.api.main.setup_logging"),
            patch("cli_agent_orchestrator.api.main.init_db"),
            patch(
                "cli_agent_orchestrator.services.memory_reconciliation.reconcile_memory_startup",
                return_value=None,
            ),
            patch("cli_agent_orchestrator.api.main.cleanup_old_data"),
            patch(
                "cli_agent_orchestrator.api.main.cleanup_expired_memories", new_callable=AsyncMock
            ),
            patch("cli_agent_orchestrator.api.main.flow_daemon", fake_flow_daemon),
            patch("cli_agent_orchestrator.api.main.bus.set_loop"),
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
            patch.object(PluginRegistry, "teardown", mock_teardown),
            patch(
                "cli_agent_orchestrator.api.main.drain_all_terminal_input_locks",
                return_value=[],
            ) as mock_drain,
        ):
            async with lifespan(app):
                mock_drain.assert_not_called()

            # Called exactly once, during shutdown, with a positive bounded
            # timeout -- never "assumed drained" without actually asking.
            mock_drain.assert_called_once()
            (timeout_arg,), _ = mock_drain.call_args
            assert timeout_arg > 0

    @pytest.mark.asyncio
    async def test_shutdown_logs_a_truthful_warning_when_a_terminal_stays_busy(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-empty drain result must be logged, never silently treated
        as drained (message-1038's own "do not invent an assertion").
        """
        (
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
        ) = _consumer_patches()

        with (
            patch("cli_agent_orchestrator.api.main.setup_logging"),
            patch("cli_agent_orchestrator.api.main.init_db"),
            patch(
                "cli_agent_orchestrator.services.memory_reconciliation.reconcile_memory_startup",
                return_value=None,
            ),
            patch("cli_agent_orchestrator.api.main.cleanup_old_data"),
            patch(
                "cli_agent_orchestrator.api.main.cleanup_expired_memories", new_callable=AsyncMock
            ),
            patch("cli_agent_orchestrator.api.main.flow_daemon", fake_flow_daemon),
            patch("cli_agent_orchestrator.api.main.bus.set_loop"),
            status_run,
            log_run,
            inbox_run,
            assigned_completion_run,
            assigned_completion_register,
            opencode_daemon,
            patch.object(PluginRegistry, "teardown", AsyncMock()),
            patch(
                "cli_agent_orchestrator.api.main.drain_all_terminal_input_locks",
                return_value=["stuck-terminal-xyz"],
            ),
        ):
            with caplog.at_level(logging.WARNING, logger="cli_agent_orchestrator.api.main"):
                async with lifespan(app):
                    pass

        assert "stuck-terminal-xyz" in caplog.text
