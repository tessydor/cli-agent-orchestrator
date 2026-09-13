"""Tests for the real DB-backed native-dispatch corruption-record mutation
(correction-994; ``clients.database.record_native_dispatch_corruption``).

The pure guard/analysis logic it delegates to is tested independently,
without a database, in test/services/test_native_dispatch_recovery.py. This
file exercises the actual transaction: fresh reads, guard rerun, and
enforced-by-the-database insert-only uniqueness, against a real, synthetic,
file-backed SQLite database and genuine concurrent connections/transactions
-- never a test-only in-process lock standing in for the database's own CAS.

All content here is synthetic fixture data invented for these tests --
nothing from any real assignment, terminal, or archive is copied into this
repository.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignmentLifecycle,
    CompletionReceiverState,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services.native_dispatch_recovery import (
    GUARD_CALLER_MISMATCH,
    GUARD_CHANGED_CALLBACK_STATE,
    GUARD_CONTENT_MISMATCH,
    GUARD_EVIDENCE_MUTATION,
    GUARD_HASH_PREFIX_MISMATCH,
    GUARD_LIVE_TERMINAL_ACTIVITY,
    GUARD_NOT_FOUND,
    GUARD_WRONG_DIRECTION,
    utf8_sha256,
)

BOUND_PREFIX = "synthetic assigned task: please review the fixture"
APPENDED_SUFFIX = "synthetic unrelated follow-up appended with no turn boundary"
CORRUPTED_TRANSCRIPT = BOUND_PREFIX + APPENDED_SUFFIX


@pytest.fixture
def corruption_db(tmp_path, monkeypatch):
    """Real, file-backed SQLite -- required (not :memory:) so genuinely
    concurrent connections from different threads see the same database.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'native-dispatch-corruption.sqlite'}",
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


def _terminal(terminal_id: str) -> None:
    db.create_terminal(terminal_id, "cao-test", f"window-{terminal_id}", "mock_cli")


def _assignment(
    worker_id: str,
    caller_id: str,
    *,
    assignment_id: str = "assignment-0001",
    completion_id: str = "completion-0001",
):
    """A real caller + a real dispatched assigned-worker callback row."""
    _terminal(caller_id)
    db.create_terminal(
        worker_id,
        "cao-test",
        f"window-{worker_id}",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=assignment_id,
        completion_id=completion_id,
    )
    record = db.mark_assigned_worker_dispatched(worker_id)
    assert record is not None
    return record


def _record_kwargs(**overrides) -> dict:
    kwargs = dict(
        assignment_id="assignment-0001",
        requesting_caller_id="synthetic-caller",
        live_status=TerminalStatus.IDLE,
        transcript_first_user_text=CORRUPTED_TRANSCRIPT,
        expected_transcript_sha256=utf8_sha256(CORRUPTED_TRANSCRIPT),
        admissible_dispatch_sha256=[utf8_sha256(BOUND_PREFIX)],
        concatenated_message_id="874",
        concatenated_message_sender_id="synthetic-caller",
        concatenated_message_receiver_id="synthetic-worker",
        concatenated_message_content=APPENDED_SUFFIX,
        concatenated_message_delivery_state="delivered",
        concatenated_message_session_id="synthetic-session",
        recorded_at="2026-09-13T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return kwargs


class TestRecordNativeDispatchCorruptionHappyPath:
    def test_first_call_persists_a_matching_durable_row(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")

        record = db.record_native_dispatch_corruption(**_record_kwargs())

        assert record.assignment_id == "assignment-0001"
        assert record.completion_id == "completion-0001"
        assert record.worker_terminal_id == "synthetic-worker"
        assert record.caller_id == "synthetic-caller"
        assert record.registered_dispatch_sha256 == utf8_sha256(BOUND_PREFIX)
        assert record.bound_prefix_length == len(BOUND_PREFIX)
        assert record.concatenated_message_content_sha256 == utf8_sha256(APPENDED_SUFFIX)

        with db.SessionLocal() as session:
            row = (
                session.query(db.NativeDispatchCorruptionModel)
                .filter(db.NativeDispatchCorruptionModel.record_key == record.record_key())
                .first()
            )
        assert row is not None
        assert row.concatenated_message_content_sha256 == utf8_sha256(APPENDED_SUFFIX)


class TestRecordNativeDispatchCorruptionGuards:
    """Every guard is rerun fresh, from the real DB row, inside the
    transaction -- not trusted from any caller-supplied precomputed result.
    """

    def test_unknown_assignment_fails_closed(self, corruption_db):
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(**_record_kwargs())
        assert exc_info.value.reason_code == GUARD_NOT_FOUND

    def test_wrong_requesting_caller_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(requesting_caller_id="someone-else")
            )
        assert exc_info.value.reason_code == GUARD_CALLER_MISMATCH

    def test_tampered_transcript_digest_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(expected_transcript_sha256="0" * 64)
            )
        assert exc_info.value.reason_code == GUARD_EVIDENCE_MUTATION

    def test_no_admissible_dispatch_hash_matches_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(admissible_dispatch_sha256=["0" * 64])
            )
        assert exc_info.value.reason_code == GUARD_HASH_PREFIX_MISMATCH

    @pytest.mark.parametrize(
        "status", [TerminalStatus.PROCESSING, TerminalStatus.WAITING_USER_ANSWER]
    )
    def test_live_terminal_fails_closed(self, corruption_db, status):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(**_record_kwargs(live_status=status))
        assert exc_info.value.reason_code == GUARD_LIVE_TERMINAL_ACTIVITY

    def test_changed_lifecycle_since_dispatch_fails_closed(self, corruption_db):
        """The callback is re-read FRESH inside the transaction -- a
        lifecycle change made after any earlier analysis is honored, not a
        stale snapshot."""
        _assignment("synthetic-worker", "synthetic-caller")
        # Move the assignment to a terminal, ineligible lifecycle before the
        # corruption record is ever attempted.
        failed = db.mark_completion_terminal_error(
            "assignment-0001",
            "synthetic terminal failure",
            CompletionReceiverState.UNKNOWN,
            lifecycle=AssignmentLifecycle.FAILED,
        )
        assert failed is not None
        assert failed.lifecycle == AssignmentLifecycle.FAILED
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(**_record_kwargs())
        assert exc_info.value.reason_code == GUARD_CHANGED_CALLBACK_STATE

    def test_wrong_content_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(concatenated_message_content="a completely different message")
            )
        assert exc_info.value.reason_code == GUARD_CONTENT_MISMATCH

    def test_wrong_sender_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(concatenated_message_sender_id="not-the-caller")
            )
        assert exc_info.value.reason_code == GUARD_WRONG_DIRECTION

    def test_wrong_receiver_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(concatenated_message_receiver_id="not-the-worker")
            )
        assert exc_info.value.reason_code == GUARD_WRONG_DIRECTION

    def test_reversed_direction_worker_to_caller_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(
                    concatenated_message_sender_id="synthetic-worker",
                    concatenated_message_receiver_id="synthetic-caller",
                )
            )
        assert exc_info.value.reason_code == GUARD_WRONG_DIRECTION

    def test_wrong_assignment_id_fails_closed(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(assignment_id="assignment-does-not-exist")
            )
        assert exc_info.value.reason_code == GUARD_NOT_FOUND

    def test_wrong_callback_via_mismatched_other_assignment_fails_closed(self, corruption_db):
        """A claim correctly shaped for assignment-0001 must not validate
        against a DIFFERENT real assignment/callback."""
        _assignment("synthetic-worker", "synthetic-caller")
        _assignment(
            "other-worker",
            "other-caller",
            assignment_id="assignment-other",
            completion_id="completion-other",
        )
        with pytest.raises(db.NativeDispatchCorruptionGuardError) as exc_info:
            db.record_native_dispatch_corruption(
                **_record_kwargs(
                    assignment_id="assignment-other",
                    requesting_caller_id="other-caller",
                )
            )
        # The claim's sender/receiver (synthetic-caller/synthetic-worker)
        # don't match assignment-other's real caller/worker pair.
        assert exc_info.value.reason_code == GUARD_WRONG_DIRECTION


class TestRecordNativeDispatchCorruptionIdempotencyAndConflict:
    def test_byte_identical_replay_is_a_safe_no_op(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        first = db.record_native_dispatch_corruption(**_record_kwargs())
        second = db.record_native_dispatch_corruption(**_record_kwargs(recorded_at="LATER"))

        assert first.record_key() == second.record_key()
        with db.SessionLocal() as session:
            count = (
                session.query(db.NativeDispatchCorruptionModel)
                .filter(db.NativeDispatchCorruptionModel.record_key == first.record_key())
                .count()
            )
        assert count == 1, "a byte-identical (modulo recorded_at) replay must never duplicate"

    def test_conflicting_content_for_the_same_identity_is_refused(self, corruption_db):
        _assignment("synthetic-worker", "synthetic-caller")
        db.record_native_dispatch_corruption(**_record_kwargs())

        # Same assignment + same concatenated_message_id (so same record_key)
        # but different content -- this requires a SECOND admissible hash so
        # the guard layer itself doesn't refuse it first for an unrelated
        # reason; the point of this test is the DB-level conflict specifically.
        other_prefix = "a different, but also admissible, bound prefix"
        other_suffix = "an entirely different concatenated message body"
        with pytest.raises(ValueError, match="collision"):
            db.record_native_dispatch_corruption(
                **_record_kwargs(
                    transcript_first_user_text=other_prefix + other_suffix,
                    expected_transcript_sha256=utf8_sha256(other_prefix + other_suffix),
                    admissible_dispatch_sha256=[utf8_sha256(other_prefix)],
                    concatenated_message_content=other_suffix,
                )
            )

    def test_concurrent_identical_calls_produce_exactly_one_row(self, corruption_db):
        """Genuine concurrent transactions from different threads/connections
        against the real database -- the row's own PRIMARY KEY is what
        prevents a duplicate, not any test-side lock."""
        _assignment("synthetic-worker", "synthetic-caller")

        def call():
            return db.record_native_dispatch_corruption(**_record_kwargs())

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(call) for _ in range(8)]
            results = [f.result(timeout=5) for f in futures]

        assert all(r.record_key() == results[0].record_key() for r in results)
        with db.SessionLocal() as session:
            count = session.query(db.NativeDispatchCorruptionModel).count()
        assert count == 1, f"expected exactly one durable row, got {count}"

    def test_concurrent_conflicting_calls_exactly_one_wins_rest_refused(self, corruption_db):
        """Same assignment/message identity, genuinely different content,
        raced across real threads/connections: exactly one must durably win
        and every other caller must be refused -- never a silent overwrite."""
        _assignment("synthetic-worker", "synthetic-caller")

        def call(variant: int):
            # Each variant's OWN transcript/claim are mutually consistent
            # (so every one independently passes every guard) -- they only
            # collide at the DB layer, on the SAME concatenated_message_id
            # under the SAME assignment (record_key is identity-based, not
            # content-based).
            suffix = f"variant {variant} of the concatenated message body"
            transcript = BOUND_PREFIX + suffix
            return db.record_native_dispatch_corruption(
                **_record_kwargs(
                    transcript_first_user_text=transcript,
                    expected_transcript_sha256=utf8_sha256(transcript),
                    concatenated_message_content=suffix,
                )
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(call, i) for i in range(6)]
            outcomes = []
            for f in futures:
                try:
                    outcomes.append(("ok", f.result(timeout=5)))
                except ValueError as e:
                    outcomes.append(("refused", str(e)))

        wins = [o for o in outcomes if o[0] == "ok"]
        refusals = [o for o in outcomes if o[0] == "refused"]
        assert len(wins) == 1, f"expected exactly one winner, got {len(wins)}: {wins}"
        assert len(refusals) == 5
        assert all("collision" in detail for _, detail in refusals)
        with db.SessionLocal() as session:
            count = session.query(db.NativeDispatchCorruptionModel).count()
        assert count == 1


class TestRecordNativeDispatchCorruptionNeverTouchesCompletionSemantics:
    def test_recording_does_not_change_lifecycle_delivery_state_or_final_result(
        self, corruption_db
    ):
        _assignment("synthetic-worker", "synthetic-caller")

        before = db.get_assigned_worker_callback("synthetic-worker")
        db.record_native_dispatch_corruption(**_record_kwargs())
        after = db.get_assigned_worker_callback("synthetic-worker")

        assert after.lifecycle == before.lifecycle
        assert after.delivery_state == before.delivery_state
        assert after.final_result == before.final_result is None
