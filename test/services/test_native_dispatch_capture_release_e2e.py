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

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
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


class TestGuardedCaptureReleaseDoesNotDeadlockAgainstAConcurrentWaiter:
    """Correction-1007 item 1: a concurrent InboxService/send_input-style
    caller that already holds terminal_input_lock(worker_id) and is
    blocked inside wait_for_capture_before_input(), waiting on the SAME
    barrier this transition exists to release, must not prevent this
    transition from completing. Uses the REAL terminal_input_lock and the
    REAL wait_for_capture_before_input to reproduce/prove this
    deterministically -- no sleeps as the proof (only bounded
    future.result() timeouts and real threading.Event synchronization).
    """

    def test_concurrent_holder_blocked_on_the_same_barrier_does_not_starve_the_release(
        self, capture_release_db, monkeypatch
    ):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        lock = real_terminal_service.terminal_input_lock(WORKER_ID)
        thread_a_ready = threading.Event()

        def thread_a() -> bool:
            # Mirrors the exact real shape of send_input / InboxService.
            # deliver_pending: acquire terminal_input_lock for this
            # worker, THEN call wait_for_capture_before_input while still
            # holding it. The 20s timeout is deliberately much larger
            # than any bounded future.result() below is given -- on
            # unfixed code this thread would not release the lock again
            # until this wait actually times out or is genuinely
            # notified, so a short future.result() on thread B below is
            # the deterministic, sleep-free proof of the hang.
            with lock:
                thread_a_ready.set()
                return service.wait_for_capture_before_input(WORKER_ID, timeout=20.0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(thread_a)
            assert thread_a_ready.wait(timeout=2), "thread A did not reach its wait in time"

            future_b = pool.submit(_apply, service)
            # On 53b031b6, reconcile_corrupted_dispatch_capture_release
            # needs this SAME terminal_input_lock, which thread A holds
            # while blocked -- this call would not return within any
            # short bound. The fix must let it complete almost
            # immediately, without depending on thread A's own timeout.
            result = future_b.result(timeout=3)
            assert result.released_now is True

            # Thread A must then be released promptly too -- well before
            # its own 20s timeout -- proving the barrier was genuinely
            # released (not merely that thread B failed closed and gave
            # up without releasing anything).
            assert future_a.result(timeout=3) is True


class TestGuardedCaptureReleaseSurvivesACrashBetweenCommitAndRelease:
    """Correction-1007 item 2: a process that dies after the DB
    acknowledgement durably commits but before the in-memory capture
    barrier is released must not strand that barrier forever. Uses the
    real fault-injection seam (_capture_release_checkpoint) to simulate
    the crash deterministically, then proves a plain retry of the exact
    same acknowledgement -- an idempotent DB replay, released_now=False --
    still releases the barrier, and that ordinary InboxService delivery
    then delivers the existing pending 948 analogue exactly once, without
    replaying 874, creating any duplicate row, or marking the corrupted
    task successful/retirement-eligible.
    """

    def test_retry_after_simulated_crash_releases_the_stranded_barrier(
        self, capture_release_db, monkeypatch
    ):
        service = _seed_incident(monkeypatch)
        monkeypatch.setattr(
            status_monitor_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE
        )

        def _simulate_crash(worker_terminal_id, result):
            raise RuntimeError("simulated process crash after DB commit, before barrier release")

        # A scoped patch.object context manager (not monkeypatch.setattr)
        # so it reverts on its own at the `with` block's end -- calling
        # monkeypatch.undo() here would also undo capture_release_db's own
        # SessionLocal patch, since fixtures and the test body share one
        # monkeypatch instance.
        with patch.object(service, "_capture_release_checkpoint", _simulate_crash):
            # The DB acknowledgement commits (acknowledge_native_dispatch_
            # corruption's own transaction has already returned/committed
            # by the time the checkpoint above runs), but the simulated
            # crash propagates out of this call before
            # _release_capture_barrier ever executes -- the barrier is
            # durably acknowledged in the DB, yet stranded armed in
            # memory, exactly the scenario item 2 describes.
            with pytest.raises(RuntimeError, match="simulated process crash"):
                _apply(service)
        assert (
            service.wait_for_capture_before_input(WORKER_ID, timeout=0.2) is False
        ), "barrier must be stranded armed immediately after the simulated crash"
        with db.SessionLocal() as session:
            row = (
                session.query(db.NativeDispatchCorruptionModel)
                .filter(db.NativeDispatchCorruptionModel.assignment_id == ASSIGNMENT_ID)
                .first()
            )
        assert (
            row is not None and row.capture_release_acknowledged_at is not None
        ), "the DB acknowledgement itself must have durably committed despite the crash"

        # Retry with the IDENTICAL acknowledgement -- an idempotent DB
        # replay (released_now=False), yet it must still release the
        # now-stranded barrier. The real (no-op) checkpoint is back in
        # effect now that the `with patch.object(...)` block above exited.
        retry = _apply(service)
        assert (
            retry.released_now is False
        ), "the DB side is an idempotent replay, not a fresh transition"
        assert (
            service.wait_for_capture_before_input(WORKER_ID, timeout=1.0) is True
        ), "the retry must release the barrier despite released_now=False"

        # The corrupted task stays not-successful; 874 is never replayed;
        # no duplicate row was created by either the crashed attempt or
        # the retry.
        after = db.get_assigned_worker_callback(WORKER_ID)
        assert after.lifecycle == AssignmentLifecycle.DISPATCHED
        assert after.final_result is None
        assert after.delivery_state == CompletionDeliveryState.NOT_READY
        all_rows = db.get_inbox_messages(WORKER_ID, limit=10)
        assert len(all_rows) == 2, "no duplicate row from the crashed attempt or the retry"
        matching_874 = [r for r in all_rows if r.message == MESSAGE_874_CONTENT]
        assert len(matching_874) == 1

        # Ordinary InboxService delivery then delivers the existing
        # pending 948 exactly once, through the real send_input path.
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
        delivered_948 = [
            r
            for r in db.get_inbox_messages(WORKER_ID, limit=10, status=MessageStatus.DELIVERED)
            if r.message == MESSAGE_948_CONTENT
        ]
        assert len(delivered_948) == 1


RESTART_CALLER_ID = "restart-caller-e2e"
RESTART_WORKER_ID = "restart-worker-e2e"
RESTART_ASSIGNMENT_ID = "assignment-restart-e2e-0001"
RESTART_COMPLETION_ID = "completion-restart-e2e-0001"
RESTART_MESSAGE_948_CONTENT = "synthetic later follow-up (948 analogue) queued before a restart"

IDLE_HOLD_CALLER_ID = "idle-hold-caller-e2e"
IDLE_HOLD_WORKER_ID = "idle-hold-worker-e2e"
IDLE_HOLD_ASSIGNMENT_ID = "assignment-idle-hold-e2e-0001"
IDLE_HOLD_COMPLETION_ID = "completion-idle-hold-e2e-0001"
IDLE_HOLD_MESSAGE_948_CONTENT = (
    "synthetic later follow-up (948 analogue) queued before a restart into IDLE"
)


class TestRestartReconstructsTheCaptureHoldBeforeConsumersStart:
    """Correction-1033: register_persisted_assignments()'s OWN docstring
    already promised "Prime restart barriers before status/inbox
    consumers can observe readiness" -- but its actual body only calls
    register_assignment() (marks the worker KNOWN), never anything that
    arms a capture barrier. The ONLY other path that arms one
    (announce_terminal_status) is called exclusively from StatusMonitor's
    live, chunk-driven detection pipeline, which never fires for a worker
    whose pane has been quiet since before the restart (the corrupted/
    uncaptured turn's own output already fully arrived). Meanwhile
    status_monitor.get_status()'s OWN restart-status derivation (its
    "cached == UNKNOWN -> probe the full history" branch, deliberately
    left UNCACHED) can independently report this SAME worker as COMPLETED
    to InboxService's separate, unrelated readiness gate -- without ever
    updating _last_status or triggering announce_terminal_status. The two
    paths disagreeing is exactly the race: InboxService believes the
    worker is ready to receive its next queued message, while nothing
    ever told AssignedWorkerCompletionService a barrier was needed.

    This reproduces the full shape through the REAL singleton
    (register_persisted_assignments and the capture-barrier check are
    both keyed off it in production code, not an injectable instance) and
    the REAL InboxService.deliver_pending -> terminal_service.send_input
    path -- deliberately never calling announce_terminal_status directly,
    unlike the pre-existing (and, per this finding, misleadingly named)
    test_restart_primes_all_unfinished_capture_barriers, which papers over
    this exact gap by calling announce_terminal_status itself right after
    register_persisted_assignments().
    """

    @staticmethod
    def _seed(worker_id, caller_id, assignment_id, completion_id, pending_content):
        db.create_terminal(caller_id, "cao-session", f"developer-{caller_id}", "mock_cli")
        db.create_terminal(
            worker_id,
            "cao-session",
            f"developer-{worker_id}",
            "mock_cli",
            caller_id=caller_id,
            assignment_id=assignment_id,
            completion_id=completion_id,
        )
        dispatched = db.mark_assigned_worker_dispatched(worker_id)
        assert dispatched is not None
        assert dispatched.lifecycle == AssignmentLifecycle.DISPATCHED
        # The existing pending 948 analogue -- queued before the simulated
        # restart, exactly as it would be in the real incident.
        db.create_inbox_message(
            caller_id, worker_id, pending_content, origin=InboxMessageOrigin.EXPLICIT
        )

    @staticmethod
    def _restart_alone(monkeypatch, assigned_worker_completion_service):
        # Simulate a fresh process: reset the REAL singleton's in-memory
        # state (mirrors test/conftest.py's own _reset_status_monitor_state
        # pattern for the sibling singleton) -- nothing is "known" yet, no
        # barrier exists, exactly like a just-started cao-server.
        assigned_worker_completion_service.__init__()
        # The ONE synchronous startup step, run alone -- deliberately NEVER
        # calling announce_terminal_status directly, since a real restart
        # has no guarantee a live detection chunk will ever arrive for a
        # quiet, already-finished pane.
        assigned_worker_completion_service.register_persisted_assignments()
        # InboxService's OWN restart-status derivation reports COMPLETED --
        # exactly what status_monitor.get_status()'s real "cached ==
        # UNKNOWN -> probe full history" branch can do, independent of
        # announce_terminal_status.
        monkeypatch.setattr(
            status_monitor_mod.status_monitor,
            "get_status",
            lambda _id: TerminalStatus.COMPLETED,
        )

    @staticmethod
    def _deliver(worker_id) -> list[str]:
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
            InboxService().deliver_pending(worker_id)
        return writes

    def test_restart_alone_does_not_let_inbox_deliver_before_the_ack(
        self, capture_release_db, monkeypatch
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        try:
            self._seed(
                RESTART_WORKER_ID,
                RESTART_CALLER_ID,
                RESTART_ASSIGNMENT_ID,
                RESTART_COMPLETION_ID,
                RESTART_MESSAGE_948_CONTENT,
            )
            self._restart_alone(monkeypatch, assigned_worker_completion_service)

            writes = self._deliver(RESTART_WORKER_ID)

            assert writes == [], (
                "InboxService delivered the pending 948 analogue to a worker whose "
                "corrupted/unresolved persisted assignment was never reconciled after "
                "a simulated restart -- correction-1033's exact race reproduced"
            )
            pending_after = db.get_pending_messages(RESTART_WORKER_ID, limit=10)
            assert len(pending_after) == 1, "the message must remain PENDING, not lost"

            # The barrier IS armed (this is the actual fix): a direct
            # capture-wait call proves it, independent of InboxService's
            # own eager-return-on-no-barrier fast path.
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    RESTART_WORKER_ID, timeout=0.2
                )
                is False
            ), "register_persisted_assignments() did not reconstruct the capture hold"
        finally:
            assigned_worker_completion_service._release_capture_barrier(RESTART_WORKER_ID)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(RESTART_WORKER_ID)

    def test_post_ack_ordinary_delivery_exactly_once(self, capture_release_db, monkeypatch):
        """After the restart-reconstructed hold blocks delivery, the
        existing guarded recovery acknowledgement (unchanged, correction-
        997/1001/1007) releases it, and ordinary InboxService delivery
        then delivers the existing pending 948 analogue exactly once --
        proving the fix does not strand the hold forever, only until the
        real, already-tested release path runs.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        bound_prefix = "synthetic restart-scenario dispatch (1033)"
        message_874_content = "synthetic already-delivered follow-up (874 analogue), restart case"
        corrupted_transcript = bound_prefix + message_874_content

        try:
            self._seed(
                RESTART_WORKER_ID,
                RESTART_CALLER_ID,
                RESTART_ASSIGNMENT_ID,
                RESTART_COMPLETION_ID,
                RESTART_MESSAGE_948_CONTENT,
            )
            self._restart_alone(monkeypatch, assigned_worker_completion_service)

            # Blocked pre-ack (same proof as the sibling test above).
            assert self._deliver(RESTART_WORKER_ID) == []

            result = assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                RESTART_WORKER_ID,
                assignment_id=RESTART_ASSIGNMENT_ID,
                requesting_caller_id=RESTART_CALLER_ID,
                transcript_first_user_text=corrupted_transcript,
                expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                concatenated_message_id="874-restart",
                concatenated_message_sender_id=RESTART_CALLER_ID,
                concatenated_message_receiver_id=RESTART_WORKER_ID,
                concatenated_message_content=message_874_content,
                concatenated_message_delivery_state="delivered",
                concatenated_message_session_id="synthetic-restart-session",
                recorded_at="2026-09-14T00:00:00+00:00",
                acknowledgement=f"{RESTART_CALLER_ID} acknowledges the corrupted dispatch is abandoned",
                acknowledged_at="2026-09-14T00:01:00+00:00",
            )
            assert result.released_now is True

            after = db.get_assigned_worker_callback(RESTART_WORKER_ID)
            assert after.lifecycle == AssignmentLifecycle.DISPATCHED
            assert after.final_result is None

            writes = self._deliver(RESTART_WORKER_ID)
            assert writes == [RESTART_MESSAGE_948_CONTENT]
            delivered = [
                r
                for r in db.get_inbox_messages(
                    RESTART_WORKER_ID, limit=10, status=MessageStatus.DELIVERED
                )
                if r.message == RESTART_MESSAGE_948_CONTENT
            ]
            assert len(delivered) == 1

            # A repeat delivery attempt is a genuine no-op (nothing left pending).
            assert self._deliver(RESTART_WORKER_ID) == []
        finally:
            assigned_worker_completion_service._release_capture_barrier(RESTART_WORKER_ID)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(RESTART_WORKER_ID)

    def test_unrelated_resolved_worker_is_unaffected(self, capture_release_db, monkeypatch):
        """A worker whose assignment is ALREADY fully resolved (lifecycle
        outside ASSIGNED/DISPATCHED/UNRESOLVED, so excluded from
        list_protected_assigned_worker_callbacks entirely) never gets a
        hold reconstructed for it, and its own delivery proceeds exactly
        as before this fix -- the change is scoped to unresolved
        assignments only, never a global weakening or strengthening of
        the "no barrier -> proceed" fast path.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        resolved_worker = "resolved-worker-e2e"
        resolved_caller = "resolved-caller-e2e"
        resolved_content = "synthetic follow-up for an already-resolved worker"
        try:
            db.create_terminal(
                resolved_caller, "cao-session", "developer-resolved-caller", "mock_cli"
            )
            db.create_terminal(
                resolved_worker,
                "cao-session",
                "developer-resolved-worker",
                "mock_cli",
                caller_id=resolved_caller,
                assignment_id="assignment-resolved-e2e",
                completion_id="completion-resolved-e2e",
            )
            dispatched = db.mark_assigned_worker_dispatched(resolved_worker)
            assert dispatched is not None
            report = "already delivered final report"
            captured = db.capture_assigned_worker_completion(
                resolved_worker,
                report,
                utf8_sha256(report),
                f"assigned-worker-callback:{dispatched.assignment_id}",
            )
            assert captured is not None
            assert captured.lifecycle == AssignmentLifecycle.COMPLETED
            inbox_msg = db.create_inbox_message(
                resolved_worker,
                resolved_caller,
                AssignedWorkerCompletionService._format_callback_message(captured),
                origin=InboxMessageOrigin.SERVER_COMPLETION,
                assignment_id=captured.assignment_id,
                idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
            )
            db.mark_completion_enqueued(
                captured.assignment_id, inbox_msg.id, CompletionReceiverState.ACTIVE
            )
            db.acknowledge_completion_enqueued(captured.assignment_id, inbox_msg.id)

            db.create_inbox_message(
                resolved_caller,
                resolved_worker,
                resolved_content,
                origin=InboxMessageOrigin.EXPLICIT,
            )

            self._restart_alone(monkeypatch, assigned_worker_completion_service)

            # Not "known" at all -- register_persisted_assignments never
            # touched it, since its lifecycle is COMPLETED, not in
            # (ASSIGNED, DISPATCHED, UNRESOLVED).
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    resolved_worker, timeout=0.2
                )
                is True
            )

            writes = self._deliver(resolved_worker)
            assert writes == [resolved_content], (
                "an unrelated, already-resolved worker's delivery must proceed exactly "
                "as before this fix -- the reconstruction is scoped to unresolved "
                "assignments only"
            )
        finally:
            assigned_worker_completion_service._release_capture_barrier(resolved_worker)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(resolved_worker)


class TestRestartHoldGovernsIdleToo:
    """Correction-1038: correction-1033's restart-reconstructed hold only
    ever got consulted by ``InboxService._deliver_pending_locked`` and
    ``terminal_service.send_input`` on the ``COMPLETED`` branch of each
    function -- both gated the capture-hold wait behind
    ``status == TerminalStatus.COMPLETED``. A worker whose native provider
    legitimately reports IDLE after a restart (a ready, boxed composer,
    with nothing new to detect) skipped that check ENTIRELY: the
    reconstructed hold existed in memory but nothing ever asked it whether
    it was safe to proceed. This reproduces that exact gap -- deliberately
    the IDLE branch, never COMPLETED -- through the same REAL
    register_persisted_assignments -> InboxService.deliver_pending ->
    terminal_service.send_input path as the sibling class above, and
    proves the fix (moving the capture-hold check outside any
    status-specific gate, in both functions) closes it without requiring
    ANY particular status label.
    """

    @staticmethod
    def _restart_alone_idle(monkeypatch, assigned_worker_completion_service):
        """Same restart simulation as the COMPLETED sibling, except the
        restart-status derivation reports IDLE -- a ready, boxed composer,
        not a fresh completion. This is the exact status label message
        1038 identifies as the one the pre-fix code never checked a hold
        against.
        """
        assigned_worker_completion_service.__init__()
        assigned_worker_completion_service.register_persisted_assignments()
        monkeypatch.setattr(
            status_monitor_mod.status_monitor,
            "get_status",
            lambda _id: TerminalStatus.IDLE,
        )

    def test_restart_into_idle_does_not_let_inbox_deliver_before_the_ack(
        self, capture_release_db, monkeypatch
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        try:
            TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._seed(
                IDLE_HOLD_WORKER_ID,
                IDLE_HOLD_CALLER_ID,
                IDLE_HOLD_ASSIGNMENT_ID,
                IDLE_HOLD_COMPLETION_ID,
                IDLE_HOLD_MESSAGE_948_CONTENT,
            )
            self._restart_alone_idle(monkeypatch, assigned_worker_completion_service)

            writes = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                IDLE_HOLD_WORKER_ID
            )

            assert writes == [], (
                "InboxService delivered the pending 948 analogue to a worker reporting "
                "IDLE whose corrupted/unresolved persisted assignment was never "
                "reconciled after a simulated restart -- correction-1038's exact gap "
                "reproduced (the hold existed but was never consulted on the IDLE "
                "branch)"
            )
            pending_after = db.get_pending_messages(IDLE_HOLD_WORKER_ID, limit=10)
            assert len(pending_after) == 1, "the message must remain PENDING, not lost"

            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    IDLE_HOLD_WORKER_ID, timeout=0.2
                )
                is False
            ), "the reconstructed hold must still be armed, independent of status"
        finally:
            assigned_worker_completion_service._release_capture_barrier(IDLE_HOLD_WORKER_ID)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(IDLE_HOLD_WORKER_ID)

    def test_post_ack_ordinary_delivery_exactly_once_from_idle(
        self, capture_release_db, monkeypatch
    ):
        """After the restart-reconstructed hold blocks delivery to an
        IDLE-reporting worker, the existing guarded recovery
        acknowledgement (unchanged, correction-997/1001/1007) releases it,
        and ordinary InboxService delivery then delivers the existing
        pending 948 analogue exactly once -- proving the IDLE-branch fix
        does not strand the hold forever, only until the real,
        already-tested release path runs.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        bound_prefix = "synthetic idle-restart-scenario dispatch (1038)"
        message_874_content = (
            "synthetic already-delivered follow-up (874 analogue), idle-restart case"
        )
        corrupted_transcript = bound_prefix + message_874_content

        try:
            TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._seed(
                IDLE_HOLD_WORKER_ID,
                IDLE_HOLD_CALLER_ID,
                IDLE_HOLD_ASSIGNMENT_ID,
                IDLE_HOLD_COMPLETION_ID,
                IDLE_HOLD_MESSAGE_948_CONTENT,
            )
            self._restart_alone_idle(monkeypatch, assigned_worker_completion_service)

            # Blocked pre-ack (same proof as the test above), still on IDLE.
            assert (
                TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                    IDLE_HOLD_WORKER_ID
                )
                == []
            )

            result = (
                assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                    IDLE_HOLD_WORKER_ID,
                    assignment_id=IDLE_HOLD_ASSIGNMENT_ID,
                    requesting_caller_id=IDLE_HOLD_CALLER_ID,
                    transcript_first_user_text=corrupted_transcript,
                    expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                    admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                    concatenated_message_id="874-idle-restart",
                    concatenated_message_sender_id=IDLE_HOLD_CALLER_ID,
                    concatenated_message_receiver_id=IDLE_HOLD_WORKER_ID,
                    concatenated_message_content=message_874_content,
                    concatenated_message_delivery_state="delivered",
                    concatenated_message_session_id="synthetic-idle-restart-session",
                    recorded_at="2026-09-14T00:00:00+00:00",
                    acknowledgement=(
                        f"{IDLE_HOLD_CALLER_ID} acknowledges the corrupted dispatch is abandoned"
                    ),
                    acknowledged_at="2026-09-14T00:01:00+00:00",
                )
            )
            assert result.released_now is True

            after = db.get_assigned_worker_callback(IDLE_HOLD_WORKER_ID)
            assert after.lifecycle == AssignmentLifecycle.DISPATCHED
            assert after.final_result is None

            # Status is STILL IDLE (never flipped to COMPLETED) -- the release
            # must be honored on the IDLE branch too, not only COMPLETED.
            writes = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                IDLE_HOLD_WORKER_ID
            )
            assert writes == [IDLE_HOLD_MESSAGE_948_CONTENT]
            delivered = [
                r
                for r in db.get_inbox_messages(
                    IDLE_HOLD_WORKER_ID, limit=10, status=MessageStatus.DELIVERED
                )
                if r.message == IDLE_HOLD_MESSAGE_948_CONTENT
            ]
            assert len(delivered) == 1

            assert (
                TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                    IDLE_HOLD_WORKER_ID
                )
                == []
            )
        finally:
            assigned_worker_completion_service._release_capture_barrier(IDLE_HOLD_WORKER_ID)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(IDLE_HOLD_WORKER_ID)

    def test_genuine_first_assigned_dispatch_is_unaffected(self, capture_release_db, monkeypatch):
        """A brand-new worker -- ``register_assignment`` has marked it
        "known" (as ``create_terminal`` does immediately for every
        assignment, real code, correction-1033's own report section 1),
        but no barrier has ever been armed for it, since neither a live
        COMPLETED detection nor a restart reconstruction has ever run for
        this fresh assignment. The now-unconditional capture-hold check
        must be a true no-op here: ``_wait_for_capture_before_input_
        detailed`` fast-paths on "no barrier exists" regardless of status,
        so a genuine first dispatch proceeds exactly as before this fix.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        fresh_worker = "fresh-dispatch-worker-e2e"
        fresh_caller = "fresh-dispatch-caller-e2e"
        try:
            db.create_terminal(fresh_caller, "cao-session", "developer-fresh-caller", "mock_cli")
            db.create_terminal(
                fresh_worker,
                "cao-session",
                "developer-fresh-worker",
                "mock_cli",
                caller_id=fresh_caller,
                assignment_id="assignment-fresh-e2e-0001",
                completion_id="completion-fresh-e2e-0001",
            )
            assigned_worker_completion_service.register_assignment(fresh_worker)
            monkeypatch.setattr(
                status_monitor_mod.status_monitor,
                "get_status",
                lambda _id: TerminalStatus.IDLE,
            )

            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    fresh_worker, timeout=0.2
                )
                is True
            ), "a brand-new worker must never have a barrier armed against it"

            writes = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                fresh_worker
            )
            # No pending message was ever queued for this worker; the point of
            # this test is solely that reaching this far raises nothing and
            # blocks nothing -- deliver_pending is simply a no-op on an empty
            # inbox, exactly as it always was.
            assert writes == []
        finally:
            assigned_worker_completion_service._release_capture_barrier(fresh_worker)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(fresh_worker)


QUEUE_CALLER_ID = "queue-caller-e2e"
QUEUE_WORKER_ID = "queue-worker-e2e"
QUEUE_ASSIGNMENT_ID = "assignment-queue-e2e-0001"
QUEUE_COMPLETION_ID = "completion-queue-e2e-0001"


class TestGuardedReconciliationSupersedesTheObsoleteQueue:
    """Correction-1038 Part B: the real B queue prerequisite. The live
    pending order is (877, 880, 885, 948) analogues -- oldest-first
    selection means an unguarded release would send 877 first, even
    though 948 is what actually supersedes them. Proves the guarded
    recovery transition durably supersedes the exact obsolete rows
    BEFORE releasing the barrier, so ordinary InboxService delivery then
    selects the ORIGINAL 948 row, physically, exactly once -- never a
    fabricated/duplicate/requeued row, never touching 874's own already-
    delivered record.
    """

    def test_supersession_prerequisite_then_exactly_once_948_delivery(
        self, capture_release_db, monkeypatch
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        bound_prefix = "synthetic queue-scenario dispatch (1038 part B)"
        message_874_content = "synthetic already-delivered follow-up (874 analogue), queue case"
        corrupted_transcript = bound_prefix + message_874_content

        try:
            db.create_terminal(
                QUEUE_CALLER_ID, "cao-session", f"developer-{QUEUE_CALLER_ID}", "mock_cli"
            )
            db.create_terminal(
                QUEUE_WORKER_ID,
                "cao-session",
                f"developer-{QUEUE_WORKER_ID}",
                "mock_cli",
                caller_id=QUEUE_CALLER_ID,
                assignment_id=QUEUE_ASSIGNMENT_ID,
                completion_id=QUEUE_COMPLETION_ID,
            )
            dispatched = db.mark_assigned_worker_dispatched(QUEUE_WORKER_ID)
            assert dispatched is not None

            # The real live pending order: three obsolete follow-ups (877/880/
            # 885 analogues), then the one that actually supersedes them (948
            # analogue) -- created in that exact order, oldest first.
            m877 = db.create_inbox_message(
                QUEUE_CALLER_ID, QUEUE_WORKER_ID, "877 analogue", origin=InboxMessageOrigin.EXPLICIT
            )
            m880 = db.create_inbox_message(
                QUEUE_CALLER_ID, QUEUE_WORKER_ID, "880 analogue", origin=InboxMessageOrigin.EXPLICIT
            )
            m885 = db.create_inbox_message(
                QUEUE_CALLER_ID, QUEUE_WORKER_ID, "885 analogue", origin=InboxMessageOrigin.EXPLICIT
            )
            m948 = db.create_inbox_message(
                QUEUE_CALLER_ID, QUEUE_WORKER_ID, "948 analogue", origin=InboxMessageOrigin.EXPLICIT
            )

            # Restart into IDLE, exactly like correction-1038's Part A fix --
            # both gaps are the same live incident.
            assigned_worker_completion_service.__init__()
            assigned_worker_completion_service.register_persisted_assignments()
            monkeypatch.setattr(
                status_monitor_mod.status_monitor,
                "get_status",
                lambda _id: TerminalStatus.IDLE,
            )

            # Pre-ack: nothing at all is deliverable (877 would otherwise be
            # selected first, oldest-first).
            assert (
                TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(QUEUE_WORKER_ID)
                == []
            )

            result = assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                QUEUE_WORKER_ID,
                assignment_id=QUEUE_ASSIGNMENT_ID,
                requesting_caller_id=QUEUE_CALLER_ID,
                transcript_first_user_text=corrupted_transcript,
                expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                concatenated_message_id="874-queue",
                concatenated_message_sender_id=QUEUE_CALLER_ID,
                concatenated_message_receiver_id=QUEUE_WORKER_ID,
                concatenated_message_content=message_874_content,
                concatenated_message_delivery_state="delivered",
                concatenated_message_session_id="synthetic-queue-session",
                recorded_at="2026-09-14T00:00:00+00:00",
                acknowledgement=f"{QUEUE_CALLER_ID} acknowledges the corrupted dispatch is abandoned",
                acknowledged_at="2026-09-14T00:01:00+00:00",
                supersede_message_ids=[m877.id, m880.id, m885.id],
                superseding_message_id=m948.id,
            )
            assert result.released_now is True

            # 877/880/885 are durably SUPERSEDED -- preserved, not deleted,
            # not fabricated DELIVERED, not requeued/duplicated.
            all_rows = {m.id: m for m in db.get_inbox_messages(QUEUE_WORKER_ID, limit=10)}
            for obsolete_id, expected_content in (
                (m877.id, "877 analogue"),
                (m880.id, "880 analogue"),
                (m885.id, "885 analogue"),
            ):
                row = all_rows[obsolete_id]
                assert row.status == MessageStatus.SUPERSEDED
                assert row.superseded_by_message_id == m948.id
                assert row.message == expected_content
                assert row.sender_id == QUEUE_CALLER_ID
                assert row.receiver_id == QUEUE_WORKER_ID
            assert all_rows[m948.id].status == MessageStatus.PENDING

            # 874 itself was never touched -- no inbox row for it exists in
            # this scenario at all (it is bound purely by transcript/hash
            # evidence, exactly like the sibling restart tests).

            # Ordinary InboxService delivery now selects the ORIGINAL 948 row
            # -- physically, exactly once -- never 877 (the oldest-first
            # candidate an unguarded release would have sent).
            writes = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                QUEUE_WORKER_ID
            )
            assert writes == ["948 analogue"]
            delivered = [
                r
                for r in db.get_inbox_messages(
                    QUEUE_WORKER_ID, limit=10, status=MessageStatus.DELIVERED
                )
                if r.id == m948.id
            ]
            assert len(delivered) == 1

            # A repeat delivery attempt is a genuine no-op.
            assert (
                TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(QUEUE_WORKER_ID)
                == []
            )
        finally:
            assigned_worker_completion_service._release_capture_barrier(QUEUE_WORKER_ID)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(QUEUE_WORKER_ID)

    def test_wrong_supersession_target_fails_closed_and_barrier_stays_armed(
        self, capture_release_db, monkeypatch
    ):
        """If the supersession prerequisite fails after the acknowledgement
        already committed, the barrier must stay armed -- no partial
        release, no stranded incomplete state presented as resolved.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        bound_prefix = "synthetic queue-scenario dispatch (1038 part B, wrong target)"
        message_874_content = "synthetic already-delivered follow-up (874 analogue), wrong target"
        corrupted_transcript = bound_prefix + message_874_content
        worker_id = "queue-worker-wrong-target-e2e"
        caller_id = "queue-caller-wrong-target-e2e"

        try:
            db.create_terminal(caller_id, "cao-session", f"developer-{caller_id}", "mock_cli")
            db.create_terminal(
                worker_id,
                "cao-session",
                f"developer-{worker_id}",
                "mock_cli",
                caller_id=caller_id,
                assignment_id="assignment-wrong-target-e2e-0001",
                completion_id="completion-wrong-target-e2e-0001",
            )
            db.mark_assigned_worker_dispatched(worker_id)
            m948 = db.create_inbox_message(
                caller_id, worker_id, "948 analogue", origin=InboxMessageOrigin.EXPLICIT
            )

            assigned_worker_completion_service.__init__()
            assigned_worker_completion_service.register_persisted_assignments()
            monkeypatch.setattr(
                status_monitor_mod.status_monitor,
                "get_status",
                lambda _id: TerminalStatus.IDLE,
            )

            with pytest.raises(ValueError, match="does not exist"):
                assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                    worker_id,
                    assignment_id="assignment-wrong-target-e2e-0001",
                    requesting_caller_id=caller_id,
                    transcript_first_user_text=corrupted_transcript,
                    expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                    admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                    concatenated_message_id="874-wrong-target",
                    concatenated_message_sender_id=caller_id,
                    concatenated_message_receiver_id=worker_id,
                    concatenated_message_content=message_874_content,
                    concatenated_message_delivery_state="delivered",
                    concatenated_message_session_id="synthetic-wrong-target-session",
                    recorded_at="2026-09-14T00:00:00+00:00",
                    acknowledgement=f"{caller_id} acknowledges the corrupted dispatch is abandoned",
                    acknowledged_at="2026-09-14T00:01:00+00:00",
                    # Nonexistent target id -- supersede_inbox_messages must
                    # refuse this, and that refusal must propagate here.
                    supersede_message_ids=[999999],
                    superseding_message_id=m948.id,
                )

            # The barrier must still be armed: release never happened.
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    worker_id, timeout=0.2
                )
                is False
            )
            # 948 itself was never touched by the failed supersession attempt.
            row = db.get_inbox_messages(worker_id, limit=10)[0]
            assert row.status == MessageStatus.PENDING
            assert row.superseded_by_message_id is None
        finally:
            assigned_worker_completion_service._release_capture_barrier(worker_id)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(worker_id)


class TestPersistedAssignedAndDurableCompletionAcrossRestart:
    """Correction-1042/1044: two confirmed defects in
    register_persisted_assignments' unrefined loop over
    list_protected_assigned_worker_callbacks' broader (ASSIGNED,
    DISPATCHED, UNRESOLVED) filter.

    1. ASSIGNED (never dispatched) must never be armed -- a hold exists to
       protect an already-captured/in-flight completion, which cannot
       exist before the assignment's first real dispatch. The existing
       test_genuine_first_assigned_dispatch_is_unaffected never actually
       went through register_persisted_assignments, so it could not catch
       this.
    2. A DISPATCHED/UNRESOLVED record whose incident has ALREADY durably
       completed its entire guarded transition must not be re-armed on a
       SECOND restart -- caller_acknowledgement alone is not sufficient
       durable proof of that; capture_release_completed_at is.
    """

    @staticmethod
    def _send_direct(worker_id: str, message: str, orchestration_type=None) -> list[str]:
        """Same mocking as TestRestartReconstructsTheCaptureHoldBefore
        ConsumersStart._deliver, but calls terminal_service.send_input
        directly -- the shape a genuine FIRST assignment dispatch takes
        (agent_step.run_agent_step), never via InboxService (which only
        ever delivers an existing PENDING row; a never-dispatched ASSIGNED
        worker has none).
        """
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
            real_terminal_service.send_input(
                worker_id, message, orchestration_type=orchestration_type
            )
        return writes

    def test_persisted_assigned_callback_is_not_armed_and_first_dispatch_is_permitted(
        self, capture_release_db, monkeypatch
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        # bind_completion_dispatch's identity guard requires an 8-hex
        # terminal_id and a 32-hex completion_id for the ASSIGN branch
        # (real production IDs always are) -- every other id in this test
        # file that never exercises that branch can use a descriptive
        # string instead.
        worker_id = "a1a1a1a1"
        caller_id = "b2b2b2b2"
        completion_id = "c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3"
        try:
            db.create_terminal(caller_id, "cao-session", f"developer-{caller_id}", "mock_cli")
            db.create_terminal(
                worker_id,
                "cao-session",
                f"developer-{worker_id}",
                "mock_cli",
                caller_id=caller_id,
                assignment_id="assignment-persisted-assigned-e2e-0001",
                completion_id=completion_id,
            )
            # Deliberately never call mark_assigned_worker_dispatched --
            # the callback row's lifecycle stays ASSIGNED, the exact
            # persisted-but-never-dispatched shape this fix targets.
            callback = db.get_assigned_worker_callback(worker_id)
            assert callback.lifecycle == AssignmentLifecycle.ASSIGNED

            # Simulate a restart with this ASSIGNED record already persisted.
            assigned_worker_completion_service.__init__()
            assigned_worker_completion_service.register_persisted_assignments()
            monkeypatch.setattr(
                status_monitor_mod.status_monitor,
                "get_status",
                lambda _id: TerminalStatus.IDLE,
            )

            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    worker_id, timeout=0.2
                )
                is True
            ), "a never-dispatched ASSIGNED record must never have a hold armed against it"

            writes = self._send_direct(
                worker_id, "first assignment dispatch", orchestration_type=OrchestrationType.ASSIGN
            )
            assert writes == ["first assignment dispatch"], (
                "the genuine first dispatch to a persisted ASSIGNED worker was blocked -- "
                "correction-1042/1044's exact regression reproduced"
            )
        finally:
            assigned_worker_completion_service._release_capture_barrier(worker_id)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(worker_id)

    def test_second_restart_does_not_rearm_a_fully_resolved_incident(
        self, capture_release_db, monkeypatch
    ):
        """Deterministic restart E2E, per message 1044's own required
        coverage: IDLE pre-ack blocks; acknowledgement + supersession
        completes and releases; original 948 delivered exactly once;
        a SECOND restart must NOT re-arm this same incident.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        worker_id = "second-restart-worker-e2e"
        caller_id = "second-restart-caller-e2e"
        assignment_id = "assignment-second-restart-e2e-0001"
        completion_id = "completion-second-restart-e2e-0001"
        message_948_content = "synthetic 948 analogue, second-restart case"
        bound_prefix = "synthetic second-restart dispatch (1042/1044)"
        message_874_content = "synthetic 874 analogue, second-restart case"
        corrupted_transcript = bound_prefix + message_874_content

        def restart(status: TerminalStatus) -> None:
            assigned_worker_completion_service.__init__()
            assigned_worker_completion_service.register_persisted_assignments()
            monkeypatch.setattr(status_monitor_mod.status_monitor, "get_status", lambda _id: status)

        try:
            TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._seed(
                worker_id, caller_id, assignment_id, completion_id, message_948_content
            )

            # First restart, into IDLE: blocked pre-ack.
            restart(TerminalStatus.IDLE)
            assert (
                TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(worker_id) == []
            )

            result = (
                assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                    worker_id,
                    assignment_id=assignment_id,
                    requesting_caller_id=caller_id,
                    transcript_first_user_text=corrupted_transcript,
                    expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                    admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                    concatenated_message_id="874-second-restart",
                    concatenated_message_sender_id=caller_id,
                    concatenated_message_receiver_id=worker_id,
                    concatenated_message_content=message_874_content,
                    concatenated_message_delivery_state="delivered",
                    concatenated_message_session_id="synthetic-second-restart-session",
                    recorded_at="2026-09-14T00:00:00+00:00",
                    acknowledgement=f"{caller_id} acknowledges the corrupted dispatch is abandoned",
                    acknowledged_at="2026-09-14T00:01:00+00:00",
                )
            )
            assert result.released_now is True

            # 948 delivered exactly once.
            writes = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(worker_id)
            assert writes == [message_948_content]

            # SECOND restart: the incident is now fully resolved (no
            # supersession was requested, so the completion marker was set
            # immediately after acknowledgement) -- must NOT re-arm.
            restart(TerminalStatus.IDLE)
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    worker_id, timeout=0.2
                )
                is True
            ), "a fully-resolved incident was re-armed on a second restart"

            # A brand-new follow-up queued after the second restart must be
            # delivered normally -- the worker is not permanently stuck.
            # The first delivery above armed the native-acceptance fence
            # (correction-984); nothing in this mocked environment ever
            # produces the real detection event that clears it, so
            # explicitly clear it here -- standing in for the genuine
            # passage of time/detection a real terminal would produce,
            # exactly like status_monitor.clear_terminal's own role in
            # every other test in this file's teardown.
            status_monitor_mod.status_monitor.clear_terminal(worker_id)
            monkeypatch.setattr(
                status_monitor_mod.status_monitor,
                "get_status",
                lambda _id: TerminalStatus.IDLE,
            )
            db.create_inbox_message(
                caller_id,
                worker_id,
                "post-second-restart follow-up",
                origin=InboxMessageOrigin.EXPLICIT,
            )
            writes_after = TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._deliver(
                worker_id
            )
            assert writes_after == ["post-second-restart follow-up"]
        finally:
            assigned_worker_completion_service._release_capture_barrier(worker_id)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(worker_id)

    def test_second_restart_still_rearms_when_supersession_never_completed(
        self, capture_release_db, monkeypatch
    ):
        """The counterpart safety case: acknowledgement alone must NOT be
        treated as durable proof of full resolution. If a requested
        supersession never completed (simulated here via the same
        fault-injection seam correction-1007 already uses), a second
        restart MUST still re-arm the hold.
        """
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        worker_id = "second-restart-incomplete-worker-e2e"
        caller_id = "second-restart-incomplete-caller-e2e"
        assignment_id = "assignment-second-restart-incomplete-e2e-0001"
        completion_id = "completion-second-restart-incomplete-e2e-0001"
        message_948_content = "synthetic 948 analogue, incomplete-supersession case"
        bound_prefix = "synthetic incomplete-supersession dispatch (1042/1044)"
        message_874_content = "synthetic 874 analogue, incomplete-supersession case"
        corrupted_transcript = bound_prefix + message_874_content

        def restart(status: TerminalStatus) -> None:
            assigned_worker_completion_service.__init__()
            assigned_worker_completion_service.register_persisted_assignments()
            monkeypatch.setattr(status_monitor_mod.status_monitor, "get_status", lambda _id: status)

        try:
            TestRestartReconstructsTheCaptureHoldBeforeConsumersStart._seed(
                worker_id, caller_id, assignment_id, completion_id, message_948_content
            )
            restart(TerminalStatus.IDLE)

            # Simulate a crash between the acknowledgement commit and the
            # completion marker: acknowledge_native_dispatch_corruption
            # itself is untouched (it genuinely succeeds and commits), but
            # mark_native_dispatch_corruption_capture_release_completed --
            # imported locally inside the method, so patching the database
            # module's own attribute is observed on the very next call --
            # is made to raise, standing in for a process crash at exactly
            # that point.
            def crash_before_marking_complete(record_key, completed_at):
                raise RuntimeError("simulated crash before completion marker")

            monkeypatch.setattr(
                db,
                "mark_native_dispatch_corruption_capture_release_completed",
                crash_before_marking_complete,
            )
            with pytest.raises(RuntimeError, match="simulated crash"):
                assigned_worker_completion_service.reconcile_corrupted_dispatch_capture_release(
                    worker_id,
                    assignment_id=assignment_id,
                    requesting_caller_id=caller_id,
                    transcript_first_user_text=corrupted_transcript,
                    expected_transcript_sha256=utf8_sha256(corrupted_transcript),
                    admissible_dispatch_sha256=[utf8_sha256(bound_prefix)],
                    concatenated_message_id="874-incomplete",
                    concatenated_message_sender_id=caller_id,
                    concatenated_message_receiver_id=worker_id,
                    concatenated_message_content=message_874_content,
                    concatenated_message_delivery_state="delivered",
                    concatenated_message_session_id="synthetic-incomplete-session",
                    recorded_at="2026-09-14T00:00:00+00:00",
                    acknowledgement=f"{caller_id} acknowledges the corrupted dispatch is abandoned",
                    acknowledged_at="2026-09-14T00:01:00+00:00",
                )

            # The barrier from the FIRST restart is still armed (the crash
            # happened after acknowledgement's own commit, but before this
            # method's own release call).
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    worker_id, timeout=0.2
                )
                is False
            )

            # SECOND restart: caller_acknowledgement IS durably set, but
            # capture_release_completed_at is NOT -- must still re-arm.
            restart(TerminalStatus.IDLE)
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    worker_id, timeout=0.2
                )
                is False
            ), "an acknowledged-but-not-fully-completed incident was NOT re-armed on restart"
        finally:
            assigned_worker_completion_service._release_capture_barrier(worker_id)
            assigned_worker_completion_service.__init__()
            status_monitor_mod.status_monitor.clear_terminal(worker_id)
