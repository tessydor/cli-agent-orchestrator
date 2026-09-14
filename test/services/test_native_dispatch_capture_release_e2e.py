"""Deterministic synthetic end-to-end regression for the guarded native-
dispatch corruption capture-release transition (correction-997/1001).

Reproduces the full incident shape through REAL persistence and REAL
service code -- a real, file-backed SQLite database, real
``clients.database`` mutations, a real ``AssignedWorkerCompletionService``
instance, a real ``InboxService`` instance, and the real
``terminal_service.send_input`` code path. Only the true hardware/tty
boundary is doubled: the tmux backend and the provider adapter, plus
``status_monitor``'s terminal-status oracle (which itself exists to
interpret raw tmux bytes this test has none of) -- exactly the same
doubling boundary every other test in this suite uses (see
test_inbox_service.py's ``test_deferred_follow_up_never_overlaps_a_
concurrent_assign_dispatch``).

All content here is synthetic fixture data invented for this test --
nothing from any real assignment, terminal, or archive is copied into this
repository.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignmentLifecycle,
    CompletionDeliveryState,
)
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin, MessageStatus, OrchestrationType
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import inbox_service as inbox_mod
from cli_agent_orchestrator.services import status_monitor as status_monitor_mod
from cli_agent_orchestrator.services import terminal_service as real_terminal_service
from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    AssignedWorkerCompletionService,
)
from cli_agent_orchestrator.services.inbox_service import InboxService
from cli_agent_orchestrator.services.native_dispatch_recovery import utf8_sha256

CALLER_ID = "synthetic-caller-e2e"
WORKER_ID = "synthetic-worker-e2e"
ASSIGNMENT_ID = "assignment-e2e-0001"
COMPLETION_ID = "completion-e2e-0001"

BOUND_PREFIX = "synthetic assigned task: please review the fixture (997/1001 e2e)"
MESSAGE_874_CONTENT = "synthetic unrelated follow-up (874 analogue) appended with no turn boundary"
CORRUPTED_TRANSCRIPT = BOUND_PREFIX + MESSAGE_874_CONTENT
MESSAGE_948_CONTENT = "synthetic later follow-up (948 analogue) still queued for the worker"

ACKNOWLEDGEMENT = "synthetic-caller-e2e acknowledges the corrupted dispatch is abandoned"
ACKNOWLEDGED_AT = "2026-09-13T02:00:00+00:00"
RECORDED_AT = "2026-09-13T01:00:00+00:00"


@pytest.fixture
def capture_release_db(tmp_path, monkeypatch):
    """Real, file-backed SQLite so every DB call below -- corruption-record
    CAS, callback reads, and inbox claim/resolve -- goes through genuine
    transactions, not an in-memory test double.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'capture-release-e2e.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        db, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    try:
        yield
    finally:
        engine.dispose()


def _seed_incident(monkeypatch) -> AssignedWorkerCompletionService:
    """Seed the full incident shape and return a fresh, isolated service
    instance (never the module singleton, so this test cannot leak state
    into any other test's capture-barrier bookkeeping).
    """
    db.create_terminal(CALLER_ID, "cao-session", "developer-caller", "mock_cli")
    db.create_terminal(
        WORKER_ID,
        "cao-session",
        "developer-worker",
        "mock_cli",
        caller_id=CALLER_ID,
        assignment_id=ASSIGNMENT_ID,
        completion_id=COMPLETION_ID,
    )
    dispatched = db.mark_assigned_worker_dispatched(WORKER_ID)
    assert dispatched is not None
    assert dispatched.lifecycle == AssignmentLifecycle.DISPATCHED

    # The already-delivered message 874 analogue: a real, historical,
    # DELIVERED inbox row, caller -> worker (the exact direction the
    # corruption mechanism requires).
    row_874 = db.create_inbox_message(
        CALLER_ID,
        WORKER_ID,
        MESSAGE_874_CONTENT,
        origin=InboxMessageOrigin.EXPLICIT,
    )
    claimed = db.claim_inbox_message(row_874.id, "e2e-claim-874")
    assert claimed is not None
    assert db.resolve_inbox_claim(row_874.id, "e2e-claim-874", MessageStatus.DELIVERED)

    # The existing pending follow-up row (948 analogue): a distinct,
    # untouched, still-PENDING message queued to the same worker.
    db.create_inbox_message(
        CALLER_ID,
        WORKER_ID,
        MESSAGE_948_CONTENT,
        origin=InboxMessageOrigin.EXPLICIT,
    )

    service = AssignedWorkerCompletionService()
    service.register_assignment(WORKER_ID)
    # StatusMonitor calls this synchronously when the (corrupted) turn's
    # COMPLETED status is first observed -- arming the capture barrier
    # without running the rest of the completion-capture pipeline, exactly
    # as production does (announce_terminal_status is deliberately
    # decoupled from handle_status_event; see its own docstring).
    service.announce_terminal_status(WORKER_ID, TerminalStatus.COMPLETED)
    return service


def _apply(service: AssignedWorkerCompletionService, **overrides):
    kwargs = dict(
        assignment_id=ASSIGNMENT_ID,
        requesting_caller_id=CALLER_ID,
        transcript_first_user_text=CORRUPTED_TRANSCRIPT,
        expected_transcript_sha256=utf8_sha256(CORRUPTED_TRANSCRIPT),
        admissible_dispatch_sha256=[utf8_sha256(BOUND_PREFIX)],
        concatenated_message_id="874",
        concatenated_message_sender_id=CALLER_ID,
        concatenated_message_receiver_id=WORKER_ID,
        concatenated_message_content=MESSAGE_874_CONTENT,
        concatenated_message_delivery_state="delivered",
        concatenated_message_session_id="synthetic-session",
        recorded_at=RECORDED_AT,
        acknowledgement=ACKNOWLEDGEMENT,
        acknowledged_at=ACKNOWLEDGED_AT,
    )
    kwargs.update(overrides)
    return service.reconcile_corrupted_dispatch_capture_release(WORKER_ID, **kwargs)


class TestGuardedCaptureReleaseEndToEnd:
    def test_full_incident_lifecycle(self, capture_release_db, monkeypatch):
        service = _seed_incident(monkeypatch)

        # 1. The capture barrier blocks BEFORE any acknowledgement.
        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

        # status_monitor is the real singleton shared with terminal_service/
        # inbox_service; mocking only its terminal-status oracle (the one
        # thing this test genuinely cannot produce without real tmux) --
        # everything else below is the real DB/service/lock code.
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        # 2. Apply WITH the exact recorded caller's acknowledgement: succeeds.
        result = _apply(service)
        assert result.released_now is True
        assert result.record.caller_acknowledgement == ACKNOWLEDGEMENT

        # 3. The barrier is now released.
        assert service.wait_for_capture_before_input(WORKER_ID, timeout=1.0) is True

        # 4. The corrupted task stays not-successful: lifecycle/delivery
        # state/final_result are exactly as before -- this transition has
        # no lifecycle/delivery/final_result columns to touch at all.
        after = db.get_assigned_worker_callback(WORKER_ID)
        assert after.lifecycle == AssignmentLifecycle.DISPATCHED
        assert after.final_result is None
        assert after.delivery_state == CompletionDeliveryState.NOT_READY

        # 5. Message 874's historical delivery is unchanged, and it was
        # never replayed (no duplicate row with its content exists).
        delivered = db.get_inbox_messages(WORKER_ID, limit=10, status=MessageStatus.DELIVERED)
        assert len(delivered) == 1
        assert delivered[0].message == MESSAGE_874_CONTENT
        assert delivered[0].sender_id == CALLER_ID
        all_rows = db.get_inbox_messages(WORKER_ID, limit=10)
        matching_874_content = [r for r in all_rows if r.message == MESSAGE_874_CONTENT]
        assert len(matching_874_content) == 1, "874 must never be replayed as a new row"
        # No new inbox row of any kind was created: exactly the two seeded
        # rows (874 delivered, 948 pending) exist.
        assert len(all_rows) == 2

        # 6. Ordinary InboxService delivery now sends the existing pending
        # 948 analogue -- through the REAL send_input code path (terminal
        # backend + provider are the only doubles).
        inbox = InboxService()
        writes = []

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

        # 7. Native acceptance observes it exactly once: the 948 row is
        # DELIVERED exactly once, never duplicated.
        pending_after = db.get_pending_messages(WORKER_ID, limit=10)
        assert pending_after == []
        delivered_948 = [
            r
            for r in db.get_inbox_messages(WORKER_ID, limit=10, status=MessageStatus.DELIVERED)
            if r.message == MESSAGE_948_CONTENT
        ]
        assert len(delivered_948) == 1

        # 8. Repeat apply/delivery is idempotent: a second identical apply
        # does not re-release (already released) and does not touch state.
        second = _apply(service)
        assert second.released_now is False
        assert second.record.record_key() == result.record.record_key()
        after_repeat = db.get_assigned_worker_callback(WORKER_ID)
        assert after_repeat.lifecycle == AssignmentLifecycle.DISPATCHED
        assert after_repeat.final_result is None

        # A second deliver_pending is a genuine no-op: nothing left PENDING.
        with (
            patch("cli_agent_orchestrator.backends.registry._backend") as mock_backend2,
            patch.object(
                real_terminal_service.provider_manager,
                "get_provider",
                return_value=provider,
            ),
        ):
            mock_backend2.send_keys.side_effect = send_keys_side_effect
            inbox.deliver_pending(WORKER_ID)
        assert writes == [MESSAGE_948_CONTENT], "repeat delivery must not duplicate the send"

    def test_wrong_caller_fails_closed_and_barrier_stays_armed(
        self, capture_release_db, monkeypatch
    ):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        with pytest.raises(db.NativeDispatchCorruptionGuardError):
            _apply(service, requesting_caller_id="not-the-recorded-caller")

        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

    def test_wrong_archive_digest_fails_closed(self, capture_release_db, monkeypatch):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        with pytest.raises(db.NativeDispatchCorruptionGuardError):
            _apply(service, expected_transcript_sha256="0" * 64)

        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

    def test_wrong_message_content_fails_closed(self, capture_release_db, monkeypatch):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        with pytest.raises(db.NativeDispatchCorruptionGuardError):
            _apply(service, concatenated_message_content="a completely different claim")

        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

    def test_live_terminal_activity_fails_closed(self, capture_release_db, monkeypatch):
        """The orchestrator reads live_status freshly INSIDE the lock -- a
        PROCESSING terminal at that moment must refuse, matching the
        real fresh read, not a stale pre-lock snapshot."""
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor,
            "get_status",
            lambda _id: TerminalStatus.PROCESSING,
        )

        with pytest.raises(db.NativeDispatchCorruptionGuardError):
            _apply(service)

        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

    def test_missing_acknowledgement_fails_closed(self, capture_release_db, monkeypatch):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        with pytest.raises(db.NativeDispatchCorruptionGuardError):
            _apply(service, acknowledgement="")

        assert service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False

    def test_concurrent_conflicting_apply_fails_closed(self, capture_release_db, monkeypatch):
        """Two callers race to acknowledge the SAME incident with genuinely
        DIFFERENT acknowledgement text: exactly one must win; the barrier
        must end up released exactly once, never twice, never left armed."""
        from concurrent.futures import ThreadPoolExecutor

        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        def call(variant: int):
            return _apply(service, acknowledgement=f"acknowledgement variant {variant}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(call, i) for i in range(4)]
            outcomes = []
            for f in futures:
                try:
                    outcomes.append(("ok", f.result(timeout=5)))
                except ValueError as e:
                    outcomes.append(("refused", str(e)))

        wins = [o for o in outcomes if o[0] == "ok"]
        refusals = [o for o in outcomes if o[0] == "refused"]
        assert len(wins) == 1, f"expected exactly one winner, got {len(wins)}: {wins}"
        assert len(refusals) == 3
        assert service.wait_for_capture_before_input(WORKER_ID, timeout=1.0) is True


class TestGuardedCaptureReleaseLockOrdering:
    """Message-1001's TOCTOU finding, proven directly: the live-status read
    must happen strictly AFTER acquiring terminal_input_lock, and the lock
    must still be held through the DB CAS -- never a live_status read taken
    before, or a lock released before, the CAS completes.
    """

    def test_lock_is_held_across_the_fresh_status_read_and_the_db_cas(
        self, capture_release_db, monkeypatch
    ):
        service = _seed_incident(monkeypatch)
        events: list[str] = []
        real_lock = __import__("threading").RLock()

        class _RecordingLock:
            def __enter__(self):
                events.append("lock_enter")
                real_lock.acquire()
                return self

            def __exit__(self, *exc):
                events.append("lock_exit")
                real_lock.release()
                return False

        def _recording_get_status(_terminal_id):
            events.append("get_status")
            return TerminalStatus.IDLE

        real_acknowledge = db.acknowledge_native_dispatch_corruption

        def _recording_acknowledge(**kwargs):
            events.append("db_cas")
            return real_acknowledge(**kwargs)

        monkeypatch.setattr(status_monitor_mod.status_monitor, "get_status", _recording_get_status)
        monkeypatch.setattr(
            real_terminal_service, "terminal_input_lock", lambda _id: _RecordingLock()
        )
        monkeypatch.setattr(
            "cli_agent_orchestrator.clients.database.acknowledge_native_dispatch_corruption",
            _recording_acknowledge,
        )

        result = _apply(service)

        assert result.released_now is True
        # The lock must be acquired BEFORE the fresh status read, and the
        # read and the DB CAS must both happen strictly between enter/exit --
        # never a read (or the CAS) outside the lock's protection.
        assert events == ["lock_enter", "get_status", "db_cas", "lock_exit"]
        # Both real_lock.acquire()/release() above ran without error: an
        # RLock double-acquire from the same thread would not raise, but a
        # missing exit would leave it locked -- assert it is free.
        assert real_lock.acquire(blocking=False) is True
        real_lock.release()
