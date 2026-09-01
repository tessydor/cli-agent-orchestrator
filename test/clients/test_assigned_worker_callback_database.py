"""Database invariants for durable assigned-worker callback delivery."""

import hashlib
import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
)
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin


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
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id="assignment-one",
        idempotency_key="completion:one",
    )
    duplicate = db.create_inbox_message(
        "22222222",
        "11111111",
        "report",
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id="assignment-one",
        idempotency_key="completion:one",
    )
    assert duplicate.id == first.id

    with pytest.raises(ValueError, match="collision"):
        db.create_inbox_message(
            "33333333",
            "11111111",
            "different report",
            origin=InboxMessageOrigin.SERVER_COMPLETION,
            assignment_id="assignment-two",
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
    assert row == ("legacy message", "legacy", None, None)
    assert indexes["uq_inbox_idempotency_key"] == 1


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

    callback = db.create_inbox_message(
        "22222222",
        "11111111",
        "server callback",
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id="assignment-link",
        idempotency_key="assigned-worker-completion:completion-link",
    )
    linked = db.mark_completion_enqueued(
        "assignment-link", callback.id, CompletionReceiverState.ACTIVE
    )
    assert linked is not None
    assert linked.delivery_state == CompletionDeliveryState.ENQUEUED

    other = db.create_inbox_message(
        "22222222",
        "11111111",
        "other callback",
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id="assignment-link",
        idempotency_key="assigned-worker-completion:other-completion",
    )
    with pytest.raises(ValueError, match="already linked"):
        db.mark_completion_enqueued("assignment-link", other.id, CompletionReceiverState.ACTIVE)
    with pytest.raises(ValueError, match="not"):
        db.acknowledge_completion_enqueued("assignment-link", other.id)

    acknowledged = db.acknowledge_completion_enqueued("assignment-link", callback.id)
    assert acknowledged is not None
    assert acknowledged.delivery_state == CompletionDeliveryState.ACKNOWLEDGED
