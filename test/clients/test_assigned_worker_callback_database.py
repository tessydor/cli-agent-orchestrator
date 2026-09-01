"""Database invariants for durable assigned-worker callback delivery."""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerIntegrityError,
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
    callback_routing_digest,
    format_server_completion_message,
)
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin, MessageStatus


@pytest.fixture
def callback_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'callback-db.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        db,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    try:
        yield
    finally:
        engine.dispose()


def test_terminal_and_assignment_identity_commit_together(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    db.create_terminal(
        "22222222",
        "cao-s",
        "worker",
        "mock_cli",
        caller_id="11111111",
        assignment_id="immutable-assignment",
        completion_id="immutable-completion",
    )

    record = db.get_assigned_worker_callback("22222222")
    assert record is not None
    assert record.assignment_id == "immutable-assignment"
    assert record.completion_id == "immutable-completion"
    assert record.worker_terminal_id == "22222222"
    assert record.caller_id == "11111111"
    assert record.lifecycle == AssignmentLifecycle.ASSIGNED


def test_partial_assignment_identity_is_rejected_without_terminal_row(callback_db):
    with pytest.raises(ValueError, match="supplied together"):
        db.create_terminal(
            "22222222",
            "cao-s",
            "worker",
            "mock_cli",
            caller_id="11111111",
            assignment_id="only-one-id",
        )
    assert db.get_terminal_metadata("22222222") is None


def test_idempotent_inbox_insert_returns_same_row_and_rejects_collision(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    first = db.create_inbox_message(
        "22222222",
        "11111111",
        "report",
        idempotency_key="completion:one",
    )
    duplicate = db.create_inbox_message(
        "22222222",
        "11111111",
        "report",
        idempotency_key="completion:one",
    )
    assert duplicate.id == first.id

    with pytest.raises(ValueError, match="collision"):
        db.create_inbox_message(
            "33333333",
            "11111111",
            "different report",
            idempotency_key="completion:one",
        )
    assert len(db.get_inbox_messages("11111111", limit=100)) == 1


def test_legacy_inbox_migration_preserves_rows_and_is_idempotent(tmp_path, monkeypatch):
    database_file = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database_file) as conn:
        conn.execute(
            "CREATE TABLE inbox ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, sender_id TEXT NOT NULL, "
            "receiver_id TEXT NOT NULL, message TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO inbox (sender_id, receiver_id, message, status) "
            "VALUES ('22222222', '11111111', 'legacy message', 'pending')"
        )
        conn.commit()
    monkeypatch.setattr(
        "cli_agent_orchestrator.constants.DATABASE_FILE", database_file, raising=False
    )

    db._migrate_inbox_callback_schema()
    db._migrate_inbox_callback_schema()

    with sqlite3.connect(database_file) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(inbox)")]
        row = conn.execute(
            "SELECT message, origin, assignment_id, idempotency_key FROM inbox"
        ).fetchone()
        indexes = {
            index[1]: index[2] for index in conn.execute("PRAGMA index_list(inbox)").fetchall()
        }
    assert columns.count("origin") == 1
    assert columns.count("assignment_id") == 1
    assert columns.count("idempotency_key") == 1
    assert columns.count("claim_token") == 1
    assert columns.count("claimed_at") == 1
    assert row == ("legacy message", "legacy", None, None)
    assert indexes["uq_inbox_idempotency_key"] == 1


def test_legacy_callback_migration_backfills_digest_and_installs_route_guard(tmp_path, monkeypatch):
    database_file = tmp_path / "legacy-callback.sqlite"
    with sqlite3.connect(database_file) as conn:
        conn.execute("CREATE TABLE terminals (id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE inbox ("
            "id INTEGER PRIMARY KEY, sender_id TEXT, receiver_id TEXT, message TEXT, "
            "status TEXT, origin TEXT, assignment_id TEXT, idempotency_key TEXT)"
        )
        conn.execute(
            "CREATE TABLE assigned_worker_callbacks ("
            "assignment_id TEXT PRIMARY KEY, completion_id TEXT, worker_terminal_id TEXT, "
            "caller_id TEXT, lifecycle TEXT, final_result TEXT, final_result_sha256 TEXT, "
            "result_reference TEXT, inbox_message_id INTEGER)"
        )
        conn.execute(
            "INSERT INTO assigned_worker_callbacks "
            "(assignment_id, completion_id, worker_terminal_id, caller_id, lifecycle) "
            "VALUES ('legacy-assignment', 'legacy-completion', '22222222', "
            "'11111111', 'assigned')"
        )
        conn.commit()
    monkeypatch.setattr(
        "cli_agent_orchestrator.constants.DATABASE_FILE", database_file, raising=False
    )

    db._migrate_assigned_worker_integrity_schema()
    db._migrate_assigned_worker_integrity_schema()

    expected_digest = callback_routing_digest(
        "legacy-assignment", "legacy-completion", "22222222", "11111111"
    )
    with sqlite3.connect(database_file) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(assigned_worker_callbacks)")]
        digest = conn.execute(
            "SELECT routing_digest FROM assigned_worker_callbacks "
            "WHERE assignment_id = 'legacy-assignment'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.DatabaseError, match="routing is immutable"):
            conn.execute(
                "UPDATE assigned_worker_callbacks SET caller_id = '33333333' "
                "WHERE assignment_id = 'legacy-assignment'"
            )

    assert columns.count("routing_digest") == 1
    assert digest == expected_digest


def test_capture_validates_digest_and_immutable_result_reference(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    db.create_terminal(
        "22222222",
        "cao-s",
        "worker",
        "mock_cli",
        caller_id="11111111",
        assignment_id="assignment-integrity",
        completion_id="completion-integrity",
    )
    db.mark_assigned_worker_dispatched("22222222")
    report = "verbatim result"
    digest = hashlib.sha256(report.encode("utf-8")).hexdigest()

    with pytest.raises(ValueError, match="does not match"):
        db.capture_assigned_worker_completion(
            "22222222", report, "incorrect-digest", "assigned-worker-callback:assignment-integrity"
        )
    with pytest.raises(ValueError, match="immutable assignment reference"):
        db.capture_assigned_worker_completion("22222222", report, digest, "wrong-reference")

    record = db.get_assigned_worker_callback("22222222")
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.DISPATCHED
    assert record.final_result is None


def test_enqueue_and_ack_reject_unrelated_or_relinked_inbox_rows(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    db.create_terminal(
        "22222222",
        "cao-s",
        "worker",
        "mock_cli",
        caller_id="11111111",
        assignment_id="assignment-link",
        completion_id="completion-link",
    )
    db.mark_assigned_worker_dispatched("22222222")
    report = "final"
    db.capture_assigned_worker_completion(
        "22222222",
        report,
        hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "assigned-worker-callback:assignment-link",
    )
    unrelated = db.create_inbox_message(
        "22222222",
        "11111111",
        "explicit message",
        origin=InboxMessageOrigin.EXPLICIT,
        assignment_id="assignment-link",
    )
    with pytest.raises(ValueError, match="immutable server completion"):
        db.mark_completion_enqueued("assignment-link", unrelated.id, CompletionReceiverState.ACTIVE)

    callback_message = format_server_completion_message(
        report,
        "22222222",
        "assignment-link",
        "completion-link",
    )
    with db.SessionLocal() as session:
        callback_row = db.InboxModel(
            sender_id="22222222",
            receiver_id="11111111",
            message=callback_message,
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id="assignment-link",
            idempotency_key="assigned-worker-completion:completion-link",
        )
        session.add(callback_row)
        session.commit()
        session.refresh(callback_row)
        callback_id = callback_row.id
    linked = db.mark_completion_enqueued(
        "assignment-link", callback_id, CompletionReceiverState.ACTIVE
    )
    assert linked is not None
    assert linked.delivery_state == CompletionDeliveryState.ENQUEUED

    with db.SessionLocal() as session:
        other_row = db.InboxModel(
            sender_id="22222222",
            receiver_id="11111111",
            message=callback_message,
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id="assignment-link",
            idempotency_key="assigned-worker-completion:other-completion",
        )
        session.add(other_row)
        session.commit()
        session.refresh(other_row)
        other_id = other_row.id
    with pytest.raises(ValueError, match="already linked"):
        db.mark_completion_enqueued("assignment-link", other_id, CompletionReceiverState.ACTIVE)
    with pytest.raises(ValueError, match="not"):
        db.acknowledge_completion_enqueued("assignment-link", other_id)

    acknowledged = db.acknowledge_completion_enqueued("assignment-link", callback_id)
    assert acknowledged is not None
    assert acknowledged.delivery_state == CompletionDeliveryState.ACKNOWLEDGED


def _captured_assignment(
    caller_id: str = "11111111",
    worker_id: str = "22222222",
    assignment_id: str = "assignment-tamper",
    completion_id: str = "completion-tamper",
    report: str = "authentic final report",
):
    db.create_terminal(caller_id, "cao-s", "caller", "mock_cli")
    db.create_terminal(
        worker_id,
        "cao-s",
        "worker",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=assignment_id,
        completion_id=completion_id,
    )
    db.mark_assigned_worker_dispatched(worker_id)
    captured = db.capture_assigned_worker_completion(
        worker_id,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        f"assigned-worker-callback:{assignment_id}",
    )
    assert captured is not None
    return captured


def test_result_tamper_is_blocked_in_sql_and_detected_again_on_read(callback_db):
    _captured_assignment()
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="result is immutable"):
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.update()
                .where(db.AssignedWorkerCallbackModel.assignment_id == "assignment-tamper")
                .values(final_result="tampered report")
            )
            session.commit()
        session.rollback()

        # Simulate storage corruption beneath the DB guard. Read validation is
        # an independent control and must still refuse the stale digest.
        session.connection().exec_driver_sql("DROP TRIGGER trg_assigned_worker_result_immutable")
        session.connection().exec_driver_sql(
            "UPDATE assigned_worker_callbacks SET final_result = 'tampered report' "
            "WHERE assignment_id = 'assignment-tamper'"
        )
        session.commit()

    with pytest.raises(AssignedWorkerIntegrityError, match="SHA-256 mismatch"):
        db.get_assigned_worker_callback("22222222")


def test_captured_result_stays_immutable_after_lifecycle_downgrade_attempt(callback_db):
    """A two-statement rewrite cannot evade the captured-result trigger."""
    _captured_assignment()
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="result is immutable"):
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.update()
                .where(db.AssignedWorkerCallbackModel.assignment_id == "assignment-tamper")
                .values(lifecycle=AssignmentLifecycle.DISPATCHED.value)
            )
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.update()
                .where(db.AssignedWorkerCallbackModel.assignment_id == "assignment-tamper")
                .values(
                    final_result="self-consistently forged report",
                    final_result_sha256=hashlib.sha256(
                        b"self-consistently forged report"
                    ).hexdigest(),
                )
            )
            session.commit()
        session.rollback()

    retained = db.get_assigned_worker_callback("22222222")
    assert retained is not None
    assert retained.lifecycle == AssignmentLifecycle.COMPLETED
    assert retained.final_result == "authentic final report"


def test_self_consistent_route_rewrite_is_rejected_by_database(callback_db):
    _captured_assignment()
    forged_digest = callback_routing_digest(
        "assignment-tamper",
        "completion-tamper",
        "22222222",
        "33333333",
    )
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="routing is immutable"):
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.update()
                .where(db.AssignedWorkerCallbackModel.assignment_id == "assignment-tamper")
                .values(caller_id="33333333", routing_digest=forged_digest)
            )
            session.commit()


def test_callback_report_audit_row_cannot_be_deleted(callback_db):
    _captured_assignment()
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="audit/report must be retained"):
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.delete().where(
                    db.AssignedWorkerCallbackModel.assignment_id == "assignment-tamper"
                )
            )
            session.commit()

    retained = db.get_assigned_worker_callback("22222222")
    assert retained is not None
    assert retained.final_result == "authentic final report"


def test_unlinked_server_inbox_crash_evidence_cannot_be_deleted(callback_db):
    captured = _captured_assignment()
    with db.SessionLocal() as session:
        inbox_row = db.InboxModel(
            sender_id=captured.worker_terminal_id,
            receiver_id=captured.caller_id,
            message=format_server_completion_message(
                captured.final_result or "",
                captured.worker_terminal_id,
                captured.assignment_id,
                captured.completion_id,
            ),
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id=captured.assignment_id,
            idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
        )
        session.add(inbox_row)
        session.commit()
        session.refresh(inbox_row)
        inbox_id = inbox_row.id

        with pytest.raises(DatabaseError, match="server inbox evidence must be retained"):
            session.execute(db.InboxModel.__table__.delete().where(db.InboxModel.id == inbox_id))
            session.commit()
        session.rollback()
        with pytest.raises(DatabaseError, match="inbox evidence is immutable"):
            session.execute(
                db.InboxModel.__table__.update()
                .where(db.InboxModel.id == inbox_id)
                .values(receiver_id="33333333", message="forged unlinked callback")
            )
            session.commit()

    messages = db.get_inbox_messages(captured.caller_id, limit=10)
    assert [message.id for message in messages] == [inbox_id]


def test_route_digest_and_impossible_state_tamper_fail_closed_on_read(callback_db):
    _captured_assignment()
    with db.SessionLocal() as session:
        session.connection().exec_driver_sql("DROP TRIGGER trg_assigned_worker_route_immutable")
        session.connection().exec_driver_sql(
            "UPDATE assigned_worker_callbacks SET caller_id = '33333333' "
            "WHERE assignment_id = 'assignment-tamper'"
        )
        session.commit()
    with pytest.raises(AssignedWorkerIntegrityError, match="routing digest mismatch"):
        db.get_assigned_worker_callback("22222222")

    # Use a distinct assignment because the first row is deliberately corrupt.
    _captured_assignment(
        caller_id="44444444",
        worker_id="55555555",
        assignment_id="assignment-state",
        completion_id="completion-state",
    )
    with db.SessionLocal() as session:
        session.connection().exec_driver_sql(
            "UPDATE assigned_worker_callbacks SET delivery_state = 'acknowledged' "
            "WHERE assignment_id = 'assignment-state'"
        )
        session.commit()
    with pytest.raises(AssignedWorkerIntegrityError, match="linked inbox evidence"):
        db.get_assigned_worker_callback("55555555")


def test_link_and_linked_payload_tamper_are_rejected(callback_db):
    captured = _captured_assignment()
    callback = db.create_inbox_message(
        captured.worker_terminal_id,
        captured.caller_id,
        format_server_completion_message(
            captured.final_result or "",
            captured.worker_terminal_id,
            captured.assignment_id,
            captured.completion_id,
        ),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=captured.assignment_id,
        idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
    )
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="inbox link is immutable"):
            session.execute(
                db.AssignedWorkerCallbackModel.__table__.update()
                .where(db.AssignedWorkerCallbackModel.assignment_id == captured.assignment_id)
                .values(inbox_message_id=None)
            )
            session.commit()
        session.rollback()
        with pytest.raises(DatabaseError, match="inbox evidence is immutable"):
            session.execute(
                db.InboxModel.__table__.update()
                .where(db.InboxModel.id == callback.id)
                .values(message="forged callback")
            )
            session.commit()


def test_concurrent_ordinary_idempotency_insert_has_one_winner(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    gate = Barrier(2)

    def create_once():
        gate.wait(timeout=2)
        return db.create_inbox_message(
            "22222222",
            "11111111",
            "one durable row",
            idempotency_key="ordinary:single-winner",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda _index: create_once(), range(2)))
    assert rows[0].id == rows[1].id
    assert len(db.get_inbox_messages("11111111", limit=100)) == 1


def test_direct_and_bulk_delete_refuse_uncaptured_assignment(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    db.create_terminal(
        "22222222",
        "cao-s",
        "worker",
        "mock_cli",
        caller_id="11111111",
        assignment_id="assignment-protected",
        completion_id="completion-protected",
    )
    assert db.delete_terminal("22222222") is False
    assert db.delete_terminals_by_session("cao-s") == 1  # caller only
    assert db.get_terminal_metadata("22222222") is not None
    assert db.get_assigned_worker_callback("22222222").lifecycle == (  # type: ignore[union-attr]
        AssignmentLifecycle.ASSIGNED
    )

    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="terminal must be retained"):
            session.query(db.TerminalModel).filter(db.TerminalModel.id == "22222222").delete(
                synchronize_session=False
            )
            session.commit()


def test_raw_receiver_delete_cannot_bypass_queued_callback_audit(callback_db):
    captured = _captured_assignment()
    db.create_inbox_message(
        captured.worker_terminal_id,
        captured.caller_id,
        format_server_completion_message(
            captured.final_result or "",
            captured.worker_terminal_id,
            captured.assignment_id,
            captured.completion_id,
        ),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=captured.assignment_id,
        idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
    )

    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="receiver deletion requires callback audit"):
            session.query(db.TerminalModel).filter(
                db.TerminalModel.id == captured.caller_id
            ).delete(synchronize_session=False)
            session.commit()

    assert db.delete_terminal(captured.caller_id) is True
    retained = db.get_assigned_worker_callback(captured.worker_terminal_id)
    assert retained is not None
    assert retained.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert retained.receiver_state == CompletionReceiverState.DELETED


def test_delivered_callback_preserves_success_and_audits_later_receiver_delete(callback_db):
    captured = _captured_assignment()
    inbox = db.create_inbox_message(
        captured.worker_terminal_id,
        captured.caller_id,
        format_server_completion_message(
            captured.final_result or "",
            captured.worker_terminal_id,
            captured.assignment_id,
            captured.completion_id,
        ),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=captured.assignment_id,
        idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
    )
    token = "delivered-before-receiver-delete"
    assert db.claim_inbox_message(inbox.id, token) is not None
    assert db.resolve_inbox_claim(inbox.id, token, MessageStatus.DELIVERED)

    # Even delivered routes use the central deletion contract so the retained
    # audit records that the receiver is now gone.
    with db.SessionLocal() as session:
        with pytest.raises(DatabaseError, match="receiver deletion requires callback audit"):
            session.query(db.TerminalModel).filter(
                db.TerminalModel.id == captured.caller_id
            ).delete(synchronize_session=False)
            session.commit()

    assert db.delete_terminal(captured.caller_id) is True
    retained = db.get_assigned_worker_callback(captured.worker_terminal_id)
    assert retained is not None
    assert retained.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
    assert retained.receiver_state == CompletionReceiverState.DELETED
    assert retained.last_error is not None
    assert "after callback paste" in retained.last_error
    assert db.get_inbox_messages(captured.caller_id, limit=10)[0].status == (
        MessageStatus.DELIVERED
    )


def test_confirmed_missing_backend_classifies_failure_before_row_delete(callback_db):
    db.create_terminal("11111111", "cao-s", "caller", "mock_cli")
    db.create_terminal(
        "22222222",
        "cao-s",
        "worker",
        "mock_cli",
        caller_id="11111111",
        assignment_id="assignment-missing",
        completion_id="completion-missing",
    )
    assert db.delete_terminal(
        "22222222",
        missing_backend=True,
        reason="synthetic backend absence proof",
    )
    assert db.get_terminal_metadata("22222222") is None
    record = db.get_assigned_worker_callback("22222222")
    assert record is not None
    assert record.lifecycle == AssignmentLifecycle.FAILED
    assert record.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert record.final_result is None
    assert record.last_error == "synthetic backend absence proof"


def test_receiver_deletion_wins_over_late_historical_enqueue_ack(callback_db):
    """A stale phase-two ack cannot resurrect a callback after caller deletion."""
    captured = _captured_assignment(
        assignment_id="assignment-delete-ack",
        completion_id="completion-delete-ack",
        report="report retained across deletion",
    )
    callback_message = format_server_completion_message(
        captured.final_result or "",
        captured.worker_terminal_id,
        captured.assignment_id,
        captured.completion_id,
    )
    with db.SessionLocal() as session:
        inbox_row = db.InboxModel(
            sender_id=captured.worker_terminal_id,
            receiver_id=captured.caller_id,
            message=callback_message,
            status=MessageStatus.PENDING.value,
            origin=InboxMessageOrigin.SERVER_COMPLETION.value,
            assignment_id=captured.assignment_id,
            idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
        )
        session.add(inbox_row)
        session.commit()
        session.refresh(inbox_row)
        inbox_id = inbox_row.id

    enqueued = db.mark_completion_enqueued(
        captured.assignment_id,
        inbox_id,
        CompletionReceiverState.ACTIVE,
    )
    assert enqueued is not None
    assert enqueued.delivery_state == CompletionDeliveryState.ENQUEUED

    assert db.delete_terminal(captured.caller_id) is True
    late_ack = db.acknowledge_completion_enqueued(captured.assignment_id, inbox_id)

    assert late_ack is not None
    assert late_ack.delivery_state == CompletionDeliveryState.TERMINAL_ERROR
    assert late_ack.receiver_state == CompletionReceiverState.DELETED
    assert late_ack.final_result == "report retained across deletion"
    assert db.get_inbox_messages(captured.caller_id, limit=10)[0].status == MessageStatus.FAILED
