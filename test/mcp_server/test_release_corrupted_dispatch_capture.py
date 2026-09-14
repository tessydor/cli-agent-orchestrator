"""MCP-to-server integration regression for the guarded native-dispatch
corruption capture-release transition's live entry point (message 1011).

Before this, ``reconcile_corrupted_dispatch_capture_release`` was defined
and tested (correction-997/1001/1007) but reachable only from an
in-process test -- no REST/MCP/CLI/server operation invoked it. This
exercises the ACTUAL supported operation exactly as a real caller would
reach it: the ``release_corrupted_dispatch_capture`` MCP tool's real
``requests`` call, routed in-process (via ``mcp_over_testclient``, see
conftest.py) into the real ``POST /assigned-workers/{id}/
corruption-recovery`` FastAPI route, into the real
``AssignedWorkerCompletionService.reconcile_corrupted_dispatch_capture_
release``, into the real ``clients.database.acknowledge_native_dispatch_
corruption`` -- against a real, isolated, file-backed SQLite database.

All content here is synthetic fixture data invented for this test --
nothing from any real assignment, terminal, or archive is copied into this
repository.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.mcp_server.server import release_corrupted_dispatch_capture
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin, MessageStatus
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import status_monitor as status_monitor_mod
from cli_agent_orchestrator.services import terminal_service as real_terminal_service
from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    assigned_worker_completion_service,
)
from cli_agent_orchestrator.services.inbox_service import InboxService
from cli_agent_orchestrator.services.native_dispatch_recovery import utf8_sha256

# CALLER_ID matches mcp_over_testclient's own fixture identity (conftest.py
# sets CAO_TERMINAL_ID="abc12345") -- release_corrupted_dispatch_capture
# always derives its own identity from that env var, never from a tool
# argument, so this is the ONE value a "correct caller" scenario can use.
CALLER_ID = "abc12345"
WRONG_CALLER_ID = "ffffffff"
WORKER_ID = "eeee1234"
ASSIGNMENT_ID = "assignment-1011-mcp"
COMPLETION_ID = "completion-1011-mcp"

BOUND_PREFIX = "synthetic assigned task via the live MCP entry point (1011)"
MESSAGE_874_CONTENT = "synthetic unrelated follow-up (874 analogue) via the live entry point"
CORRUPTED_TRANSCRIPT = BOUND_PREFIX + MESSAGE_874_CONTENT
MESSAGE_948_CONTENT = "synthetic later follow-up (948 analogue) via the live entry point"


@pytest.fixture
def isolated_db(tmp_path):
    """Point clients.database at a real, isolated, file-backed SQLite DB."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'mcp-corruption-recovery.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with patch("cli_agent_orchestrator.clients.database.SessionLocal", factory):
        yield engine


def _seed_incident(worker_id: str, caller_id: str) -> None:
    db.create_terminal(caller_id, "cao-session", f"developer-{caller_id}", "mock_cli")
    db.create_terminal(
        worker_id,
        "cao-session",
        f"developer-{worker_id}",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=ASSIGNMENT_ID,
        completion_id=COMPLETION_ID,
    )
    dispatched = db.mark_assigned_worker_dispatched(worker_id)
    assert dispatched is not None

    row_874 = db.create_inbox_message(
        caller_id, worker_id, MESSAGE_874_CONTENT, origin=InboxMessageOrigin.EXPLICIT
    )
    claimed = db.claim_inbox_message(row_874.id, "mcp-claim-874")
    assert claimed is not None
    assert db.resolve_inbox_claim(row_874.id, "mcp-claim-874", MessageStatus.DELIVERED)

    db.create_inbox_message(
        caller_id, worker_id, MESSAGE_948_CONTENT, origin=InboxMessageOrigin.EXPLICIT
    )

    assigned_worker_completion_service.register_assignment(worker_id)
    assigned_worker_completion_service.announce_terminal_status(worker_id, TerminalStatus.COMPLETED)


def _call(**overrides):
    kwargs = dict(
        worker_terminal_id=WORKER_ID,
        assignment_id=ASSIGNMENT_ID,
        transcript_first_user_text=CORRUPTED_TRANSCRIPT,
        expected_transcript_sha256=utf8_sha256(CORRUPTED_TRANSCRIPT),
        admissible_dispatch_sha256=[utf8_sha256(BOUND_PREFIX)],
        concatenated_message_id="874",
        concatenated_message_sender_id=CALLER_ID,
        concatenated_message_receiver_id=WORKER_ID,
        concatenated_message_content=MESSAGE_874_CONTENT,
        concatenated_message_delivery_state="delivered",
        concatenated_message_session_id="synthetic-session",
        recorded_at="2026-09-14T00:00:00+00:00",
        acknowledgement=f"{CALLER_ID} acknowledges the corrupted dispatch is abandoned",
        acknowledged_at="2026-09-14T00:01:00+00:00",
    )
    kwargs.update(overrides)
    return release_corrupted_dispatch_capture(**kwargs)


class TestReleaseCorruptedDispatchCaptureLiveEntryPoint:
    def test_wrong_caller_is_refused(self, mcp_over_testclient, isolated_db, monkeypatch):
        """Message 1011: authorization is bound to the assignment's
        recorded caller_id, never accepted from an arbitrary parameter.
        The MCP tool's caller identity (abc12345, from CAO_TERMINAL_ID)
        does not match this assignment's real recorded caller
        (WRONG_CALLER_ID) -- the live entry point must refuse, and the
        barrier must remain armed.
        """
        try:
            _seed_incident(WORKER_ID, WRONG_CALLER_ID)
            monkeypatch.setattr(
                status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
            )

            result = _call()

            assert result["success"] is False, result
            assert "caller_mismatch" in result["error"]
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=0.2
                )
                is False
            ), "a refused request must never release the barrier"
        finally:
            assigned_worker_completion_service._release_capture_barrier(WORKER_ID)

    def test_successful_release_then_normal_delivery(
        self, mcp_over_testclient, isolated_db, monkeypatch
    ):
        """The full successful synthetic sequence through the live entry
        point: barrier blocks before the call, the correct recorded
        caller's request releases it, the task stays not-successful, 874
        is never replayed, no duplicate row is created, and ordinary
        InboxService delivery then delivers the existing pending 948
        exactly once through the real send_input path. A repeat call
        (idempotent replay) does not release again or duplicate anything.
        """
        try:
            _seed_incident(WORKER_ID, CALLER_ID)
            monkeypatch.setattr(
                status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
            )

            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=0.2
                )
                is False
            )

            result = _call()
            assert result["success"] is True, result
            assert result["released_now"] is True
            assert result["assignment_id"] == ASSIGNMENT_ID
            assert result["worker_terminal_id"] == WORKER_ID

            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=1.0
                )
                is True
            )

            after = db.get_assigned_worker_callback(WORKER_ID)
            assert after.final_result is None
            all_rows = db.get_inbox_messages(WORKER_ID, limit=10)
            assert len(all_rows) == 2, "no new inbox row from the recovery call itself"
            matching_874 = [r for r in all_rows if r.message == MESSAGE_874_CONTENT]
            assert len(matching_874) == 1, "874 must never be replayed"

            # Idempotent replay through the SAME live entry point: no
            # re-release, no duplicate row, no conflict.
            replay = _call()
            assert replay["success"] is True, replay
            assert replay["released_now"] is False
            all_rows_after_replay = db.get_inbox_messages(WORKER_ID, limit=10)
            assert len(all_rows_after_replay) == 2

            # Ordinary InboxService delivery then delivers the existing
            # pending 948 exactly once, through the real send_input path.
            inbox = InboxService()
            writes: list[str] = []

            def send_keys_side_effect(session, window, message, **kwargs):
                writes.append(message)

            provider = MagicMock()
            provider.paste_enter_count = 1
            provider.paste_submit_delay = 0.0
            provider.accepts_input_while_processing = False
            provider.assume_processing_on_dispatch = False
            provider.blocks_orchestrated_input_while_waiting_user_answer = False
            provider.force_bracketed_paste = True
            provider.encode_terminal_input = lambda message, orchestration_value: message

            with (
                patch("cli_agent_orchestrator.backends.registry._backend") as mock_backend,
                patch.object(
                    real_terminal_service.provider_manager,
                    "get_provider",
                    return_value=provider,
                ),
            ):
                mock_backend.send_keys.side_effect = send_keys_side_effect
                mock_backend.supports_event_inbox.return_value = False
                mock_backend.session_exists.return_value = True
                mock_backend.get_history.return_value = ""
                inbox.deliver_pending(WORKER_ID)

            assert writes == [MESSAGE_948_CONTENT]
            delivered_948 = [
                r
                for r in db.get_inbox_messages(WORKER_ID, limit=10, status=MessageStatus.DELIVERED)
                if r.message == MESSAGE_948_CONTENT
            ]
            assert len(delivered_948) == 1
        finally:
            assigned_worker_completion_service._release_capture_barrier(WORKER_ID)
