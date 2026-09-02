"""Regression and state-machine tests for assigned-worker completion callbacks."""

import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import count
from threading import Barrier
from unittest.mock import AsyncMock, MagicMock

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
from cli_agent_orchestrator.services import assigned_worker_completion_service as completion_mod
from cli_agent_orchestrator.services import cleanup_service as cleanup_mod
from cli_agent_orchestrator.services import terminal_service as terminal_mod
from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    AssignedWorkerCompletionService,
)
from cli_agent_orchestrator.services.event_bus import EventBus


@pytest.fixture
def callback_db(tmp_path, monkeypatch):
    """Use a file-backed SQLite database so restart tests cross sessions."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'callbacks.db'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db, "SessionLocal", sessions)
    try:
        yield
    finally:
        engine.dispose()


@pytest.fixture
def ids():
    """Generate valid, deterministic-looking terminal IDs without collisions."""
    values = count(1)

    def _next() -> str:
        return f"{next(values):08x}"

    return _next


def _terminal(terminal_id: str) -> None:
    db.create_terminal(terminal_id, "cao-test", f"window-{terminal_id}", "mock_cli")


def _assignment(worker_id: str, caller_id: str, sequence: int = 1):
    """Create an atomic terminal+assignment record and mark its task accepted."""
    db.create_terminal(
        worker_id,
        "cao-test",
        f"window-{worker_id}",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=f"assignment-{sequence:04d}",
        completion_id=f"completion-{sequence:04d}",
    )
    record = db.mark_assigned_worker_dispatched(worker_id)
    assert record is not None
    return record


def _unproven_assignment(worker_id: str, caller_id: str, sequence: int = 1):
    """Create a persisted route whose assigned input has not been accepted."""
    db.create_terminal(
        worker_id,
        "cao-test",
        f"window-{worker_id}",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=f"assignment-{sequence:04d}",
        completion_id=f"completion-{sequence:04d}",
    )
    record = db.get_assigned_worker_callback(worker_id)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.ASSIGNED
    return record


def _configure_real_terminal_retirement(monkeypatch, tmp_path, *, cleanup_succeeds: bool):
    """Keep the real callback-aware delete path while isolating backend effects."""
    backend = MagicMock()
    backend.get_pane_working_directory.return_value = None
    backend.get_history.return_value = "synthetic pre-dispatch transcript"
    monkeypatch.setattr(terminal_mod, "get_backend", lambda: backend)
    monkeypatch.setattr(terminal_mod, "get_herdr_inbox_service", lambda: None)
    monkeypatch.setattr(terminal_mod.fifo_manager, "stop_reader", lambda _terminal: None)
    monkeypatch.setattr(terminal_mod.status_monitor, "clear_terminal", lambda _terminal: None)
    monkeypatch.setattr(
        terminal_mod.provider_manager,
        "cleanup_provider",
        lambda _terminal: cleanup_succeeds,
    )
    monkeypatch.setattr(terminal_mod, "TERMINAL_LOG_DIR", tmp_path)
    return backend


def _service(monkeypatch, report: str = "final report") -> AssignedWorkerCompletionService:
    """Build a service with deterministic output/receiver and no tmux delivery."""
    service = AssignedWorkerCompletionService()
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: report)
    monkeypatch.setattr(
        service,
        "_classify_receiver",
        lambda _record: (CompletionReceiverState.ACTIVE, None),
    )
    monkeypatch.setattr(service, "_attempt_immediate_inbox_delivery", lambda _caller: None)
    return service


def _messages(receiver_id: str):
    return db.get_inbox_messages(receiver_id, limit=100)


def test_success_without_send_message_generates_exactly_one_callback(callback_db, ids, monkeypatch):
    """Observed regression: final output exists but the model never invokes send_message."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "self-contained final response")

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    messages = _messages(caller)
    assert len(messages) == 1
    assert messages[0].origin == InboxMessageOrigin.SERVER_COMPLETION
    assert messages[0].sender_id == worker
    assert messages[0].receiver_id == caller
    assert messages[0].message.startswith("self-contained final response\n\n")

    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.COMPLETED
    assert record.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert record.receiver_state == CompletionReceiverState.ACTIVE
    assert record.final_result == "self-contained final response"
    assert (
        record.final_result_sha256 == hashlib.sha256(b"self-contained final response").hexdigest()
    )
    assert record.result_reference == f"assigned-worker-callback:{record.assignment_id}"
    assert record.attempt_count == 1
    assert record.first_attempt_at is not None
    assert record.last_attempt_at is not None
    assert record.enqueued_at is not None
    assert record.acknowledged_at is not None


def test_callback_uses_immutable_persisted_caller(callback_db, ids, monkeypatch):
    caller, unrelated, worker = ids(), ids(), ids()
    _terminal(caller)
    _terminal(unrelated)
    _assignment(worker, caller)
    service = _service(monkeypatch)

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    assert len(_messages(caller)) == 1
    assert _messages(unrelated) == []


def test_equivalent_explicit_final_callback_suppresses_server_duplicate(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    explicit = db.create_inbox_message(
        worker,
        caller,
        "final report\n\n"
        f"[Message from terminal {worker}. Use send_message MCP tool for any follow-up work.]",
        origin=InboxMessageOrigin.EXPLICIT,
    )
    assert explicit.assignment_id == assignment.assignment_id
    service = _service(monkeypatch, "final report")

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    messages = _messages(caller)
    assert [message.origin for message in messages] == [InboxMessageOrigin.EXPLICIT]
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.delivery_state == CompletionDeliveryState.SUPPRESSED_EXPLICIT
    assert record.inbox_message_id == explicit.id


def test_unrelated_intermediate_message_is_preserved_and_does_not_suppress(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    intermediate = db.create_inbox_message(
        worker,
        caller,
        "progress: tests are still running",
        origin=InboxMessageOrigin.EXPLICIT,
    )
    service = _service(monkeypatch, "different final report")

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    messages = _messages(caller)
    assert [
        message.id for message in messages if message.origin == InboxMessageOrigin.EXPLICIT
    ] == [intermediate.id]
    assert sum(message.origin == InboxMessageOrigin.SERVER_COMPLETION for message in messages) == 1


def test_duplicate_completed_events_do_not_duplicate(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch)

    service.handle_status_event(worker, TerminalStatus.COMPLETED)
    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    assert len(_messages(caller)) == 1
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.attempt_count == 1


def test_concurrent_completed_events_serialize_to_one_callback(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "one final report")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.handle_status_event, worker, TerminalStatus.COMPLETED)
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=2)

    assert len(_messages(caller)) == 1
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.attempt_count == 1


def test_completion_capture_barrier_blocks_input_until_report_is_durable():
    service = AssignedWorkerCompletionService()
    worker = "00000001"
    service.register_assignment(worker)

    service.announce_terminal_status(worker, TerminalStatus.COMPLETED)
    assert service.wait_for_capture_before_input(worker, timeout=0) is False

    service._release_capture_barrier(worker)
    assert service.wait_for_capture_before_input(worker, timeout=0) is True


def test_capture_failure_keeps_barrier_closed(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch)
    service.register_assignment(worker)
    service.announce_terminal_status(worker, TerminalStatus.COMPLETED)
    monkeypatch.setattr(
        service,
        "_capture_final_result",
        lambda _worker: (_ for _ in ()).throw(RuntimeError("transcript unavailable")),
    )

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    assert service.wait_for_capture_before_input(worker, timeout=0) is False
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.delivery_state == CompletionDeliveryState.RETRYABLE
    assert record.final_result is None


def test_restart_primes_all_unfinished_capture_barriers(callback_db, ids):
    caller, unfinished_worker, acknowledged_worker = ids(), ids(), ids()
    _terminal(caller)
    _assignment(unfinished_worker, caller, sequence=1)
    acknowledged = _assignment(acknowledged_worker, caller, sequence=2)
    report = "already delivered"
    captured = db.capture_assigned_worker_completion(
        acknowledged_worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{acknowledged.assignment_id}",
    )
    assert captured is not None
    inbox = db.create_inbox_message(
        acknowledged_worker,
        caller,
        AssignedWorkerCompletionService._format_callback_message(captured),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=acknowledged.assignment_id,
        idempotency_key=f"assigned-worker-completion:{acknowledged.completion_id}",
    )
    db.mark_completion_enqueued(
        acknowledged.assignment_id, inbox.id, CompletionReceiverState.ACTIVE
    )
    db.acknowledge_completion_enqueued(acknowledged.assignment_id, inbox.id)

    restarted = AssignedWorkerCompletionService()
    restarted.register_persisted_assignments()
    restarted.announce_terminal_status(unfinished_worker, TerminalStatus.COMPLETED)
    restarted.announce_terminal_status(acknowledged_worker, TerminalStatus.COMPLETED)

    assert restarted.wait_for_capture_before_input(unfinished_worker, timeout=0) is False
    assert restarted.wait_for_capture_before_input(acknowledged_worker, timeout=0) is True


def test_completed_status_before_dispatch_gate_is_not_task_completion(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    db.create_terminal(
        worker,
        "cao-test",
        f"window-{worker}",
        "mock_cli",
        caller_id=caller,
        assignment_id="assignment-before-dispatch",
        completion_id="completion-before-dispatch",
    )
    service = _service(monkeypatch)

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    assert _messages(caller) == []
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.ASSIGNED
    assert record.final_result is None


def test_restart_never_promotes_unproven_assignment_from_live_completed_status(
    callback_db, ids, monkeypatch
):
    """Startup chrome cannot become a successful assigned-task report."""
    caller, worker = ids(), ids()
    _terminal(caller)
    db.create_terminal(
        worker,
        "cao-test",
        f"window-{worker}",
        "mock_cli",
        caller_id=caller,
        assignment_id="assignment-unproven-restart",
        completion_id="completion-unproven-restart",
    )
    service = _service(monkeypatch, "startup chrome, not proven task output")
    monkeypatch.setattr(service, "_detect_live_status", lambda _record: TerminalStatus.COMPLETED)

    service.reconcile_worker(worker)

    assert _messages(caller) == []
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.UNRESOLVED
    assert record.delivery_state == CompletionDeliveryState.MANUAL_RECOVERY
    assert record.final_result is None
    assert "without durable dispatch proof" in (record.last_error or "")


def test_restart_recovers_genuinely_persisted_dispatched_completion(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "proven dispatched final result")
    monkeypatch.setattr(service, "_detect_live_status", lambda _record: TerminalStatus.COMPLETED)

    service.reconcile_worker(worker)

    assert [message.sender_id for message in _messages(caller)] == [worker]
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.COMPLETED
    assert record.final_result == "proven dispatched final result"


def test_restart_before_enqueue_reconciles_captured_report(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "captured before simulated process stop"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    assert captured.delivery_state == CompletionDeliveryState.CAPTURED

    restarted = _service(monkeypatch, report)
    restarted.reconcile_pending()

    assert len(_messages(caller)) == 1
    assert db.get_assigned_worker_callback(worker).delivery_state == (  # type: ignore[union-attr]
        CompletionDeliveryState.ACKNOWLEDGED
    )


@pytest.mark.parametrize("link_before_restart", [False, True])
def test_restart_after_enqueue_before_ack_reuses_exact_inbox_row(
    callback_db, ids, monkeypatch, link_before_restart
):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "durable report"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    db.record_completion_delivery_attempt(assignment.assignment_id, CompletionReceiverState.ACTIVE)
    # Reproduce the two durable crash boundaries from the earlier phased
    # implementation. New writes commit insert+link+ack atomically, but restart
    # compatibility must adopt either historical shape without duplicating.
    with db.SessionLocal() as session:
        inbox_row = db.InboxModel(
            sender_id=worker,
            receiver_id=caller,
            message=AssignedWorkerCompletionService._format_callback_message(captured),
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id=assignment.assignment_id,
            idempotency_key=f"assigned-worker-completion:{assignment.completion_id}",
        )
        session.add(inbox_row)
        session.commit()
        session.refresh(inbox_row)
        inbox_id = inbox_row.id
    if link_before_restart:
        db.mark_completion_enqueued(
            assignment.assignment_id, inbox_id, CompletionReceiverState.ACTIVE
        )

    restarted = _service(monkeypatch, report)
    restarted.reconcile_pending()

    messages = _messages(caller)
    assert [message.id for message in messages] == [inbox_id]
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert record.inbox_message_id == inbox_id


def test_deleted_receiver_is_terminal_error_but_report_is_retained(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    db.delete_terminal(caller)
    service = AssignedWorkerCompletionService()
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: "recover me")

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    assert _messages(caller) == []
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.COMPLETED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.receiver_state == CompletionReceiverState.DELETED
    assert record.final_result == "recover me"
    assert record.terminal_error_at is not None


def test_permanently_invalid_receiver_is_not_rerouted(callback_db, ids, monkeypatch):
    worker = ids()
    db.create_terminal(
        worker,
        "cao-test",
        f"window-{worker}",
        "mock_cli",
        caller_id="invalid-caller",
        assignment_id="assignment-invalid-caller",
        completion_id="completion-invalid-caller",
    )
    db.mark_assigned_worker_dispatched(worker)
    service = AssignedWorkerCompletionService()
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: "retained report")

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.receiver_state == CompletionReceiverState.PERMANENTLY_INVALID
    assert record.caller_id == "invalid-caller"
    assert record.final_result == "retained report"


def test_retained_unreachable_receiver_keeps_one_pending_callback(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch)
    monkeypatch.setattr(
        service,
        "_classify_receiver",
        lambda _record: (CompletionReceiverState.RETAINED_UNREACHABLE, None),
    )

    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    messages = _messages(caller)
    assert len(messages) == 1
    assert messages[0].status == MessageStatus.PENDING
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert record.receiver_state == CompletionReceiverState.RETAINED_UNREACHABLE


def test_retention_never_deletes_a_pending_assigned_callback(
    callback_db, ids, monkeypatch, tmp_path
):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "durable callback"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    callback = db.create_inbox_message(
        worker,
        caller,
        AssignedWorkerCompletionService._format_callback_message(captured),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=assignment.assignment_id,
        idempotency_key=f"assigned-worker-completion:{assignment.completion_id}",
    )
    ordinary = db.create_inbox_message(worker, caller, "ordinary old message")
    with db.SessionLocal() as session:
        old = datetime.now() - timedelta(days=30)
        session.query(db.InboxModel).filter(
            db.InboxModel.id.in_((callback.id, ordinary.id))
        ).update({db.InboxModel.created_at: old}, synchronize_session=False)
        session.commit()

    monkeypatch.setattr(cleanup_mod, "SessionLocal", db.SessionLocal)
    monkeypatch.setattr(cleanup_mod, "RETENTION_DAYS", 7)
    monkeypatch.setattr(cleanup_mod, "TERMINAL_LOG_DIR", tmp_path / "no-terminal-logs")
    monkeypatch.setattr(cleanup_mod, "LOG_DIR", tmp_path / "no-server-logs")

    cleanup_mod.cleanup_old_data()

    assert [message.id for message in _messages(caller)] == [callback.id]


def test_retention_preserves_unlinked_server_row_from_legacy_crash_boundary(
    callback_db, ids, monkeypatch, tmp_path
):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "durable report awaiting legacy linkage"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    with db.SessionLocal() as session:
        server_row = db.InboxModel(
            sender_id=worker,
            receiver_id=caller,
            message=AssignedWorkerCompletionService._format_callback_message(captured),
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id=assignment.assignment_id,
            idempotency_key=f"assigned-worker-completion:{assignment.completion_id}",
            created_at=datetime.now() - timedelta(days=30),
        )
        ordinary_row = db.InboxModel(
            sender_id=worker,
            receiver_id=caller,
            message="expired ordinary message",
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.LEGACY.value,
            created_at=datetime.now() - timedelta(days=30),
        )
        session.add_all((server_row, ordinary_row))
        session.commit()
        session.refresh(server_row)
        server_id = server_row.id

    monkeypatch.setattr(cleanup_mod, "SessionLocal", db.SessionLocal)
    monkeypatch.setattr(cleanup_mod, "RETENTION_DAYS", 7)
    monkeypatch.setattr(cleanup_mod, "TERMINAL_LOG_DIR", tmp_path / "no-terminal-logs")
    monkeypatch.setattr(cleanup_mod, "LOG_DIR", tmp_path / "no-server-logs")

    cleanup_mod.cleanup_old_data()

    assert [message.id for message in _messages(caller)] == [server_id]
    retained = db.get_assigned_worker_callback(worker)
    assert retained is not None
    assert retained.delivery_state == CompletionDeliveryState.CAPTURED
    assert retained.inbox_message_id is None


def test_retention_preserves_uncaptured_dispatched_worker_report_handle(
    callback_db, ids, monkeypatch, tmp_path
):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    with db.SessionLocal() as session:
        session.query(db.TerminalModel).filter(db.TerminalModel.id == worker).update(
            {db.TerminalModel.last_active: datetime.now() - timedelta(days=30)},
            synchronize_session=False,
        )
        session.commit()

    global_service = completion_mod.assigned_worker_completion_service
    monkeypatch.setattr(
        global_service,
        "_detect_live_status",
        lambda _record: TerminalStatus.COMPLETED,
    )
    monkeypatch.setattr(
        global_service,
        "_capture_final_result",
        lambda _worker: (_ for _ in ()).throw(RuntimeError("transcript capture unavailable")),
    )
    monkeypatch.setattr(cleanup_mod, "SessionLocal", db.SessionLocal)
    monkeypatch.setattr(cleanup_mod, "RETENTION_DAYS", 7)
    terminal_log_dir = tmp_path / "terminal-logs"
    terminal_log_dir.mkdir()
    protected_paths = [
        terminal_log_dir / f"{worker}.log",
        terminal_log_dir / f"{worker}.scrollback",
        terminal_log_dir / f"{worker}.snapshot.json",
    ]
    unrelated_path = terminal_log_dir / "deadbeef.log"
    old_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
    for path in (*protected_paths, unrelated_path):
        path.write_text("synthetic recovery evidence", encoding="utf-8")
        os.utime(path, (old_timestamp, old_timestamp))
    monkeypatch.setattr(cleanup_mod, "TERMINAL_LOG_DIR", terminal_log_dir)
    monkeypatch.setattr(cleanup_mod, "LOG_DIR", tmp_path / "no-server-logs")

    cleanup_mod.cleanup_old_data()

    assert db.get_terminal_metadata(worker) is not None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.DISPATCHED
    assert record.delivery_state == CompletionDeliveryState.RETRYABLE
    assert record.final_result is None
    assert "capture failed" in (record.last_error or "").lower()
    assert all(path.exists() for path in protected_paths)
    assert not unrelated_path.exists()


def test_missing_backend_provider_deferral_still_classifies_worker_failure(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    monkeypatch.setattr(terminal_mod, "get_herdr_inbox_service", lambda: None)
    monkeypatch.setattr(terminal_mod.fifo_manager, "stop_reader", lambda _terminal: None)
    monkeypatch.setattr(terminal_mod.status_monitor, "clear_terminal", lambda _terminal: None)
    monkeypatch.setattr(
        terminal_mod.provider_manager,
        "cleanup_provider",
        lambda _terminal: False,
    )

    assert (
        terminal_mod.delete_missing_terminal(
            worker,
            "synthetic positive proof that the backend pane is absent",
        )
        is False
    )

    assert db.get_terminal_metadata(worker) is not None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.FAILED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.final_result is None
    assert record.last_error == "synthetic positive proof that the backend pane is absent"


@pytest.mark.asyncio
async def test_never_started_deferred_worker_is_failed_then_really_deleted(
    callback_db, ids, monkeypatch, tmp_path
):
    """The input-not-accepted path may retire an ASSIGNED worker without lying."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _unproven_assignment(worker, caller)
    backend = _configure_real_terminal_retirement(
        monkeypatch,
        tmp_path,
        cleanup_succeeds=True,
    )
    monkeypatch.setattr(terminal_mod, "send_input", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        terminal_mod,
        "_confirm_worker_started_or_resubmit",
        AsyncMock(return_value=False),
    )
    provider = AsyncMock()
    provider.initialize.return_value = True
    provider.shell_baseline = None

    before = set(terminal_mod._deferred_init_tasks)
    terminal_mod._schedule_deferred_init(
        provider,
        worker,
        "perform the synthetic task",
        OrchestrationType.ASSIGN,
        None,
    )
    (task,) = set(terminal_mod._deferred_init_tasks) - before
    await task

    assert db.get_terminal_metadata(worker) is None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.FAILED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.final_result is None
    assert record.last_error is not None
    assert "failed before dispatch" in record.last_error
    messages = _messages(caller)
    assert len(messages) == 1
    assert messages[0].message == (
        f"Worker {worker} received the assigned task but never started processing "
        "(input not accepted after retries). It has been deleted — re-assign the task."
    )
    backend.kill_window.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_failure_is_failed_then_really_deleted(
    callback_db, ids, monkeypatch, tmp_path
):
    """An initialize exception is proven pre-input failure and permits teardown."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _unproven_assignment(worker, caller)
    _configure_real_terminal_retirement(monkeypatch, tmp_path, cleanup_succeeds=True)
    provider = AsyncMock()
    provider.initialize.side_effect = RuntimeError("synthetic init failure")

    before = set(terminal_mod._deferred_init_tasks)
    terminal_mod._schedule_deferred_init(
        provider,
        worker,
        "perform the synthetic task",
        OrchestrationType.ASSIGN,
        None,
    )
    (task,) = set(terminal_mod._deferred_init_tasks) - before
    await task

    assert db.get_terminal_metadata(worker) is None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.FAILED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.final_result is None
    assert _messages(caller)[0].message == (
        f"Worker {worker} failed to initialize: RuntimeError('synthetic init failure'). "
        "It has been deleted — re-assign the task."
    )


@pytest.mark.asyncio
async def test_input_side_effect_exception_is_unresolved_and_never_deleted(
    callback_db, ids, monkeypatch, tmp_path
):
    """An exception after paste may have started the task, so teardown fails closed."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _unproven_assignment(worker, caller)
    backend = _configure_real_terminal_retirement(
        monkeypatch,
        tmp_path,
        cleanup_succeeds=True,
    )

    def uncertain_send(*_args, **_kwargs):
        raise RuntimeError("synthetic post-paste uncertainty")

    monkeypatch.setattr(terminal_mod, "send_input", uncertain_send)
    provider = AsyncMock()
    provider.initialize.return_value = True
    provider.shell_baseline = None

    before = set(terminal_mod._deferred_init_tasks)
    terminal_mod._schedule_deferred_init(
        provider,
        worker,
        "perform the synthetic task",
        OrchestrationType.ASSIGN,
        None,
    )
    (task,) = set(terminal_mod._deferred_init_tasks) - before
    await task

    assert db.get_terminal_metadata(worker) is not None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.UNRESOLVED
    assert record.delivery_state == CompletionDeliveryState.MANUAL_RECOVERY
    assert record.final_result is None
    assert "external side effect may have begun" in (record.last_error or "")
    assert _messages(caller)[0].message == (
        f"Worker {worker} encountered RuntimeError('synthetic post-paste uncertainty') "
        "after initial input delivery began; task acceptance is unresolved. The worker "
        "terminal/report recovery handle remains for manual recovery."
    )
    backend.kill_window.assert_not_called()


def test_deferred_failure_delete_deferral_retains_failed_worker_and_reports_truth(
    callback_db, ids, monkeypatch, tmp_path
):
    """Exact cfdee34 regression: a False delete result cannot claim deletion."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _unproven_assignment(worker, caller)
    _configure_real_terminal_retirement(monkeypatch, tmp_path, cleanup_succeeds=False)

    terminal_mod._notify_caller_of_deferred_failure(
        worker,
        f"Worker {worker} failed to initialize. It has been deleted — re-assign.",
        None,
        delete_worker=True,
    )

    assert db.get_terminal_metadata(worker) is not None
    record = db.get_assigned_worker_callback(worker)
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.FAILED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.final_result is None
    messages = _messages(caller)
    assert len(messages) == 1
    assert messages[0].message == (
        f"Worker {worker} failed to initialize. Cleanup was deferred; the worker "
        "terminal/report recovery handle remains for retry or manual recovery."
    )
    assert "has been deleted" not in messages[0].message


def test_retryable_enqueue_failure_retries_once_without_duplicate(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch)
    real_create = db.create_inbox_message
    attempts = count(1)

    def flaky_create(*args, **kwargs):
        if next(attempts) == 1:
            raise RuntimeError("database temporarily busy")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(completion_mod, "create_inbox_message", flaky_create)

    service.handle_status_event(worker, TerminalStatus.COMPLETED)
    failed = db.get_assigned_worker_callback(worker)
    assert failed is not None
    assert failed.delivery_state == CompletionDeliveryState.RETRYABLE
    assert _messages(caller) == []

    service.reconcile_pending()
    assert len(_messages(caller)) == 1
    recovered = db.get_assigned_worker_callback(worker)
    assert recovered is not None
    assert recovered.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert recovered.attempt_count == 2


def test_retryable_receiver_classification_recovers_without_enqueueing_early(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch)
    classifications = iter(
        (
            (
                CompletionReceiverState.RETRYABLE_FAILURE,
                "backend metadata temporarily unavailable",
            ),
            (CompletionReceiverState.ACTIVE, None),
        )
    )
    monkeypatch.setattr(service, "_classify_receiver", lambda _record: next(classifications))

    service.handle_status_event(worker, TerminalStatus.COMPLETED)
    retryable = db.get_assigned_worker_callback(worker)
    assert retryable is not None
    assert retryable.delivery_state == CompletionDeliveryState.RETRYABLE
    assert retryable.receiver_state == CompletionReceiverState.RETRYABLE_FAILURE
    assert retryable.attempt_count == 0
    assert _messages(caller) == []

    service.reconcile_pending()

    assert len(_messages(caller)) == 1
    recovered = db.get_assigned_worker_callback(worker)
    assert recovered is not None
    assert recovered.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert recovered.receiver_state == CompletionReceiverState.ACTIVE
    assert recovered.attempt_count == 1


@pytest.mark.asyncio
async def test_enqueue_failure_recovers_automatically_without_reconcile_poll_or_restart(
    callback_db, ids, monkeypatch
):
    """A post-start enqueue failure wakes one delayed, idempotent server retry."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = AssignedWorkerCompletionService(
        retry_initial_delay=0.01,
        retry_max_delay=0.02,
    )
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: "automatic report")
    monkeypatch.setattr(
        service,
        "_classify_receiver",
        lambda _record: (CompletionReceiverState.ACTIVE, None),
    )
    monkeypatch.setattr(service, "_attempt_immediate_inbox_delivery", lambda _caller: None)
    real_create = db.create_inbox_message
    loop = asyncio.get_running_loop()
    delivered = asyncio.Event()
    create_calls = 0

    def flaky_create(*args, **kwargs):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 1:
            raise RuntimeError("database temporarily busy")
        result = real_create(*args, **kwargs)
        loop.call_soon_threadsafe(delivered.set)
        return result

    monkeypatch.setattr(completion_mod, "create_inbox_message", flaky_create)
    service.start_retry_scheduler()
    try:
        await asyncio.to_thread(service.handle_status_event, worker, TerminalStatus.COMPLETED)
        first = db.get_assigned_worker_callback(worker)
        assert first is not None
        assert first.delivery_state == CompletionDeliveryState.RETRYABLE

        # Duplicate failure/event wakeups coalesce behind the same worker key.
        for _ in range(10):
            service._request_retry(worker)
        await asyncio.wait_for(delivered.wait(), timeout=1)
    finally:
        await service.stop_retry_scheduler()

    recovered = db.get_assigned_worker_callback(worker)
    assert recovered is not None
    assert recovered.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert recovered.attempt_count == 2
    assert create_calls == 2
    assert len(_messages(caller)) == 1


@pytest.mark.asyncio
async def test_receiver_classification_failure_recovers_automatically_without_polling(
    callback_db, ids, monkeypatch
):
    """Transient receiver discovery is retried without another status edge."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = AssignedWorkerCompletionService(
        retry_initial_delay=0.01,
        retry_max_delay=0.02,
    )
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: "classified report")
    monkeypatch.setattr(service, "_attempt_immediate_inbox_delivery", lambda _caller: None)
    classifications = iter(
        (
            (
                CompletionReceiverState.RETRYABLE_FAILURE,
                "receiver backend temporarily unavailable",
            ),
            (CompletionReceiverState.ACTIVE, None),
        )
    )
    monkeypatch.setattr(service, "_classify_receiver", lambda _record: next(classifications))
    real_create = db.create_inbox_message
    loop = asyncio.get_running_loop()
    delivered = asyncio.Event()

    def signal_create(*args, **kwargs):
        result = real_create(*args, **kwargs)
        loop.call_soon_threadsafe(delivered.set)
        return result

    monkeypatch.setattr(completion_mod, "create_inbox_message", signal_create)
    service.start_retry_scheduler()
    try:
        await asyncio.to_thread(service.handle_status_event, worker, TerminalStatus.COMPLETED)
        assert _messages(caller) == []
        await asyncio.wait_for(delivered.wait(), timeout=1)
    finally:
        await service.stop_retry_scheduler()

    recovered = db.get_assigned_worker_callback(worker)
    assert recovered is not None
    assert recovered.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert recovered.receiver_state == CompletionReceiverState.ACTIVE
    assert recovered.attempt_count == 1
    assert len(_messages(caller)) == 1


@pytest.mark.asyncio
async def test_retry_scheduler_shutdown_cancels_deadlines_and_bounds_backoff(
    callback_db, ids, monkeypatch
):
    """One shared scheduler caps delay and performs no work after shutdown."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = AssignedWorkerCompletionService(
        retry_initial_delay=0.01,
        retry_max_delay=0.02,
    )
    monkeypatch.setattr(service, "_capture_final_result", lambda _worker: "retained report")
    monkeypatch.setattr(service, "_attempt_immediate_inbox_delivery", lambda _caller: None)
    loop = asyncio.get_running_loop()
    second_failure = asyncio.Event()
    second_retry_requested = asyncio.Event()
    classification_calls = 0
    retry_requests = 0

    def always_transient(_record):
        nonlocal classification_calls
        classification_calls += 1
        if classification_calls == 2:
            loop.call_soon_threadsafe(second_failure.set)
        return CompletionReceiverState.RETRYABLE_FAILURE, "still transient"

    monkeypatch.setattr(service, "_classify_receiver", always_transient)
    real_request_retry = service._request_retry

    def signal_retry_request(worker_id):
        nonlocal retry_requests
        retry_requests += 1
        real_request_retry(worker_id)
        if retry_requests == 2:
            loop.call_soon_threadsafe(second_retry_requested.set)

    monkeypatch.setattr(service, "_request_retry", signal_retry_request)
    service.start_retry_scheduler()
    scheduler = service._retry_scheduler_task
    try:
        await asyncio.to_thread(service.handle_status_event, worker, TerminalStatus.COMPLETED)
        await asyncio.wait_for(second_failure.wait(), timeout=1)
        await asyncio.wait_for(second_retry_requested.wait(), timeout=1)
        # The deadline callback was queued before the test signal above.
        await asyncio.sleep(0)
        assert service._retry_scheduler_task is scheduler
        assert len(service._retry_due) == 1
        assert service._retry_delays[worker] <= 0.02
    finally:
        await service.stop_retry_scheduler()

    calls_at_shutdown = classification_calls
    await asyncio.sleep(0.05)
    assert classification_calls == calls_at_shutdown
    assert service._retry_scheduler_task is None
    assert service._retry_due == {}
    assert service._retry_delays == {}


@pytest.mark.asyncio
async def test_run_owns_startup_reconciliation_and_cleans_retry_scheduler(
    callback_db, ids, monkeypatch
):
    """Process startup recovers RETRYABLE state without an external sweep call."""
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    before_restart = _service(monkeypatch, "startup-retained report")
    monkeypatch.setattr(
        before_restart,
        "_classify_receiver",
        lambda _record: (CompletionReceiverState.RETRYABLE_FAILURE, "pre-restart outage"),
    )
    before_restart.handle_status_event(worker, TerminalStatus.COMPLETED)
    assert (
        db.get_assigned_worker_callback(worker).delivery_state == CompletionDeliveryState.RETRYABLE
    )

    restarted = AssignedWorkerCompletionService(
        retry_initial_delay=0.01,
        retry_max_delay=0.02,
    )
    monkeypatch.setattr(
        restarted,
        "_classify_receiver",
        lambda _record: (CompletionReceiverState.ACTIVE, None),
    )
    monkeypatch.setattr(restarted, "_attempt_immediate_inbox_delivery", lambda _caller: None)
    real_create = db.create_inbox_message
    loop = asyncio.get_running_loop()
    delivered = asyncio.Event()

    def signal_create(*args, **kwargs):
        result = real_create(*args, **kwargs)
        loop.call_soon_threadsafe(delivered.set)
        return result

    monkeypatch.setattr(completion_mod, "create_inbox_message", signal_create)
    run_task = asyncio.create_task(restarted.run())
    try:
        await asyncio.wait_for(delivered.wait(), timeout=1)
    finally:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    assert restarted._retry_scheduler_task is None
    assert db.get_assigned_worker_callback(worker).delivery_state == (
        CompletionDeliveryState.ACKNOWLEDGED
    )
    assert len(_messages(caller)) == 1


def test_multiple_workers_one_caller_each_deliver_once(callback_db, ids, monkeypatch):
    caller, worker_one, worker_two = ids(), ids(), ids()
    _terminal(caller)
    _assignment(worker_one, caller, sequence=1)
    _assignment(worker_two, caller, sequence=2)
    service = _service(monkeypatch)

    service.handle_status_event(worker_one, TerminalStatus.COMPLETED)
    service.handle_status_event(worker_two, TerminalStatus.COMPLETED)

    messages = _messages(caller)
    assert len(messages) == 2
    assert {message.sender_id for message in messages} == {worker_one, worker_two}


def test_two_callers_never_cross_routes(callback_db, ids, monkeypatch):
    caller_one, caller_two, worker_one, worker_two = ids(), ids(), ids(), ids()
    _terminal(caller_one)
    _terminal(caller_two)
    _assignment(worker_one, caller_one, sequence=1)
    _assignment(worker_two, caller_two, sequence=2)
    service = _service(monkeypatch)

    service.handle_status_event(worker_one, TerminalStatus.COMPLETED)
    service.handle_status_event(worker_two, TerminalStatus.COMPLETED)

    assert [message.sender_id for message in _messages(caller_one)] == [worker_one]
    assert [message.sender_id for message in _messages(caller_two)] == [worker_two]


def test_failure_and_cancellation_never_emit_success_callback(callback_db, ids, monkeypatch):
    caller, failed_worker, cancelled_worker = ids(), ids(), ids()
    _terminal(caller)
    _assignment(failed_worker, caller, sequence=1)
    _assignment(cancelled_worker, caller, sequence=2)
    service = _service(monkeypatch)

    service.handle_status_event(failed_worker, TerminalStatus.ERROR)
    monkeypatch.setattr(service, "_detect_live_status", lambda _record: TerminalStatus.IDLE)
    assert service.prepare_terminal_retirement(cancelled_worker) is True

    assert _messages(caller) == []
    assert db.get_assigned_worker_callback(failed_worker).lifecycle == (  # type: ignore[union-attr]
        AssignmentLifecycle.FAILED
    )
    assert db.get_assigned_worker_callback(cancelled_worker).lifecycle == (  # type: ignore[union-attr]
        AssignmentLifecycle.CANCELLED
    )


def test_report_integrity_and_manual_recovery_outlive_both_terminals(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    report = "line one\nline two\nverbatim terminal result"
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, report)
    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    delivered = db.get_assigned_worker_callback(worker)
    assert delivered is not None and delivered.inbox_message_id is not None
    claim_token = "genuinely-delivered-before-delete"
    assert db.claim_inbox_message(delivered.inbox_message_id, claim_token) is not None
    assert db.resolve_inbox_claim(delivered.inbox_message_id, claim_token, MessageStatus.DELIVERED)

    db.delete_terminal(worker)
    db.delete_terminal(caller)

    retained = db.get_assigned_worker_callback(worker)
    assert retained is not None
    assert retained.final_result == report
    assert retained.final_result_sha256 == hashlib.sha256(report.encode("utf-8")).hexdigest()
    assert retained.result_reference == f"assigned-worker-callback:{retained.assignment_id}"
    assert retained.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert retained.receiver_state == CompletionReceiverState.DELETED


def test_queued_callback_becomes_deleted_receiver_manual_recovery(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "retain this queued report")
    service.handle_status_event(worker, TerminalStatus.COMPLETED)

    queued = db.get_assigned_worker_callback(worker)
    assert queued is not None and queued.inbox_message_id is not None
    assert _messages(caller)[0].status == MessageStatus.PENDING

    assert db.delete_terminal(caller) is True

    retained = db.get_assigned_worker_callback(worker)
    assert retained is not None
    assert retained.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert retained.receiver_state == CompletionReceiverState.DELETED
    assert retained.final_result == "retain this queued report"
    assert _messages(caller)[0].status == MessageStatus.FAILED


def test_server_wins_then_equivalent_explicit_send_returns_same_callback_row(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "same final")
    service.handle_status_event(worker, TerminalStatus.COMPLETED)
    server = _messages(caller)[0]

    explicit_result = db.create_inbox_message(
        worker,
        caller,
        "same final\n\n"
        f"[Message from terminal {worker}. Use send_message MCP tool for any follow-up work.]",
        origin=InboxMessageOrigin.EXPLICIT,
    )

    assert explicit_result.id == server.id
    assert [message.id for message in _messages(caller)] == [server.id]
    assert server.origin == InboxMessageOrigin.SERVER_COMPLETION


def test_concurrent_explicit_final_and_server_insert_commit_one_callback(
    callback_db, ids, monkeypatch
):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "same final"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    gate = Barrier(2)

    def explicit_send():
        gate.wait(timeout=2)
        return db.create_inbox_message(
            worker,
            caller,
            report
            + f"\n\n[Message from terminal {worker}. Use send_message MCP tool for any follow-up work.]",
            origin=InboxMessageOrigin.EXPLICIT,
        )

    def server_send():
        gate.wait(timeout=2)
        return db.create_inbox_message(
            worker,
            caller,
            AssignedWorkerCompletionService._format_callback_message(captured),
            origin=InboxMessageOrigin.SERVER_COMPLETION,
            assignment_id=assignment.assignment_id,
            idempotency_key=f"assigned-worker-completion:{assignment.completion_id}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        explicit_future = executor.submit(explicit_send)
        server_future = executor.submit(server_send)
        explicit_row = explicit_future.result(timeout=5)
        server_row = server_future.result(timeout=5)

    assert explicit_row.id == server_row.id
    assert len(_messages(caller)) == 1
    linked = db.get_assigned_worker_callback(worker)
    assert linked is not None and linked.inbox_message_id == explicit_row.id


def test_concurrent_non_equivalent_explicit_message_is_never_suppressed(callback_db, ids):
    caller, worker = ids(), ids()
    _terminal(caller)
    assignment = _assignment(worker, caller)
    report = "final report"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment.assignment_id}",
    )
    assert captured is not None
    gate = Barrier(2)

    def create_explicit():
        gate.wait(timeout=2)
        return db.create_inbox_message(
            worker,
            caller,
            "unrelated intermediate progress",
            origin=InboxMessageOrigin.EXPLICIT,
        )

    def create_server():
        gate.wait(timeout=2)
        return db.create_inbox_message(
            worker,
            caller,
            AssignedWorkerCompletionService._format_callback_message(captured),
            origin=InboxMessageOrigin.SERVER_COMPLETION,
            assignment_id=assignment.assignment_id,
            idempotency_key=f"assigned-worker-completion:{assignment.completion_id}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = [executor.submit(create_explicit), executor.submit(create_server)]
        [future.result(timeout=5) for future in rows]
    assert len(_messages(caller)) == 2


@pytest.mark.asyncio
async def test_status_event_delivers_without_supervisor_polling(callback_db, ids, monkeypatch):
    caller, worker = ids(), ids()
    _terminal(caller)
    _assignment(worker, caller)
    service = _service(monkeypatch, "event-driven result")
    local_bus = EventBus()
    local_bus.set_loop(asyncio.get_running_loop())
    monkeypatch.setattr(completion_mod, "bus", local_bus)
    delivered = asyncio.Event()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        service,
        "_attempt_immediate_inbox_delivery",
        lambda _caller: loop.call_soon_threadsafe(delivered.set),
    )
    task = asyncio.create_task(service.run())
    await asyncio.sleep(0)
    try:
        local_bus.publish(f"terminal.{worker}.status", {"status": "completed"})
        await asyncio.wait_for(delivered.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        local_bus.set_loop(None)

    messages = _messages(caller)
    assert len(messages) == 1
    assert messages[0].origin == InboxMessageOrigin.SERVER_COMPLETION
