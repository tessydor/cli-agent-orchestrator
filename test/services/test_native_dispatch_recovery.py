"""Tests for the correction-971 corrupted-native-dispatch recovery path.

All transcript/evidence content here is synthetic fixture data invented for
these tests -- nothing from any real assignment, terminal, or archive is
copied into this repository.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
)
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services.native_dispatch_recovery import (
    GUARD_CALLER_MISMATCH,
    GUARD_CHANGED_CALLBACK_STATE,
    GUARD_DURABLE_SUFFIX_UNBOUND,
    GUARD_EVIDENCE_MUTATION,
    GUARD_HASH_PREFIX_MISMATCH,
    GUARD_LIVE_TERMINAL_ACTIVITY,
    GUARD_NOT_FOUND,
    GUARD_UNBOUND_SUFFIX_MISSING,
    MAX_TRANSCRIPT_BYTES,
    analyze_native_dispatch_corruption,
    apply_recovery_inbox_message,
    build_recovery_inbox_message_params,
    check_recovery_guards,
    utf8_sha256,
)


def _callback(
    *,
    assignment_id="assignment-synthetic-1",
    completion_id="c" * 32,
    worker_terminal_id="synthetic-worker",
    caller_id="synthetic-caller",
    lifecycle=AssignmentLifecycle.DISPATCHED,
    delivery_state=CompletionDeliveryState.RETRYABLE,
) -> AssignedWorkerCallback:
    return AssignedWorkerCallback(
        assignment_id=assignment_id,
        completion_id=completion_id,
        worker_terminal_id=worker_terminal_id,
        caller_id=caller_id,
        routing_digest="d" * 32,
        lifecycle=lifecycle,
        delivery_state=delivery_state,
        receiver_state=CompletionReceiverState.ACTIVE,
        created_at=datetime.now(),
    )


class TestAnalyzeNativeDispatchCorruption:
    def test_exact_match_has_no_unbound_suffix(self):
        text = "the exact bytes bind_completion_dispatch bound"
        analysis = analyze_native_dispatch_corruption(text, [utf8_sha256(text)])
        assert analysis.matched is True
        assert analysis.bound_prefix_length == len(text)
        assert analysis.unbound_suffix is None
        assert analysis.unbound_suffix_sha256 is None

    def test_corrupted_transcript_splits_at_the_admissible_prefix(self):
        bound_prefix = "synthetic assigned task: please review the fixture"
        appended_follow_up = "synthetic unrelated follow-up appended with no turn boundary"
        corrupted = bound_prefix + appended_follow_up
        analysis = analyze_native_dispatch_corruption(corrupted, [utf8_sha256(bound_prefix)])
        assert analysis.matched is True
        assert analysis.bound_prefix_length == len(bound_prefix)
        assert analysis.matched_digest == utf8_sha256(bound_prefix)
        assert analysis.unbound_suffix == appended_follow_up
        assert analysis.unbound_suffix_sha256 == utf8_sha256(appended_follow_up)

    def test_picks_the_longest_admissible_prefix_when_multiple_admit(self):
        # Two legitimate dispatch attempts (e.g. a retry with a different
        # memory prelude) can each be admissible; the longer one is the real
        # boundary when both happen to be prefixes of the corrupted text.
        shorter = "short admissible prefix"
        longer = shorter + " plus more admissible content"
        corrupted = longer + "trailing unrelated bytes"
        analysis = analyze_native_dispatch_corruption(
            corrupted, [utf8_sha256(shorter), utf8_sha256(longer)]
        )
        assert analysis.bound_prefix_length == len(longer)
        assert analysis.unbound_suffix == "trailing unrelated bytes"

    def test_no_admissible_digest_matches_any_prefix(self):
        analysis = analyze_native_dispatch_corruption(
            "completely unrelated transcript text", ["0" * 64]
        )
        assert analysis.matched is False
        assert analysis.bound_prefix_length is None
        assert analysis.unbound_suffix is None

    def test_empty_admissible_set_never_matches(self):
        analysis = analyze_native_dispatch_corruption("anything", [])
        assert analysis.matched is False

    def test_oversized_transcript_is_refused(self):
        oversized = "x" * (MAX_TRANSCRIPT_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds"):
            analyze_native_dispatch_corruption(oversized, [utf8_sha256(oversized)])


class TestCheckRecoveryGuards:
    def _base_kwargs(self, **overrides):
        transcript_sha = "a" * 64
        kwargs = dict(
            requesting_caller_id="synthetic-caller",
            callback=_callback(),
            analysis=analyze_native_dispatch_corruption(
                "bound partsuffix", [utf8_sha256("bound part")]
            ),
            live_status=TerminalStatus.IDLE,
            archived_transcript_sha256=transcript_sha,
            expected_archived_transcript_sha256=transcript_sha,
            durable_suffix_reference="inbox-message-99",
        )
        kwargs.update(overrides)
        return kwargs

    def test_all_guards_pass(self):
        result = check_recovery_guards(**self._base_kwargs())
        assert result.allowed is True
        assert result.reason_code is None

    def test_missing_callback_record_fails_closed(self):
        result = check_recovery_guards(**self._base_kwargs(callback=None))
        assert result.allowed is False
        assert result.reason_code == GUARD_NOT_FOUND

    def test_wrong_requesting_caller_fails_closed(self):
        result = check_recovery_guards(**self._base_kwargs(requesting_caller_id="someone-else"))
        assert result.allowed is False
        assert result.reason_code == GUARD_CALLER_MISMATCH

    def test_tampered_archive_fails_closed(self):
        result = check_recovery_guards(
            **self._base_kwargs(expected_archived_transcript_sha256="b" * 64)
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_EVIDENCE_MUTATION

    def test_no_hash_match_fails_closed(self):
        result = check_recovery_guards(
            **self._base_kwargs(
                analysis=analyze_native_dispatch_corruption("unrelated text", ["0" * 64])
            )
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_HASH_PREFIX_MISMATCH

    def test_exact_uncorrupted_match_has_nothing_to_recover(self):
        exact = "already bound exactly"
        result = check_recovery_guards(
            **self._base_kwargs(
                analysis=analyze_native_dispatch_corruption(exact, [utf8_sha256(exact)])
            )
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_UNBOUND_SUFFIX_MISSING

    @pytest.mark.parametrize(
        "lifecycle",
        [AssignmentLifecycle.FAILED, AssignmentLifecycle.CANCELLED, AssignmentLifecycle.ASSIGNED],
    )
    def test_ineligible_lifecycle_fails_closed(self, lifecycle):
        result = check_recovery_guards(**self._base_kwargs(callback=_callback(lifecycle=lifecycle)))
        assert result.allowed is False
        assert result.reason_code == GUARD_CHANGED_CALLBACK_STATE

    @pytest.mark.parametrize(
        "status", [TerminalStatus.PROCESSING, TerminalStatus.WAITING_USER_ANSWER]
    )
    def test_live_terminal_activity_fails_closed(self, status):
        result = check_recovery_guards(**self._base_kwargs(live_status=status))
        assert result.allowed is False
        assert result.reason_code == GUARD_LIVE_TERMINAL_ACTIVITY

    def test_unbound_durable_reference_fails_closed_rather_than_guessing(self):
        result = check_recovery_guards(**self._base_kwargs(durable_suffix_reference=None))
        assert result.allowed is False
        assert result.reason_code == GUARD_DURABLE_SUFFIX_UNBOUND

    def test_guard_order_identity_and_evidence_checked_before_liveness(self):
        # Wrong caller AND corrupted archive AND currently live all at once --
        # the caller-identity guard must win; nothing downstream is trusted
        # once identity itself is unverified.
        result = check_recovery_guards(
            **self._base_kwargs(
                requesting_caller_id="someone-else",
                expected_archived_transcript_sha256="mismatched",
                live_status=TerminalStatus.PROCESSING,
            )
        )
        assert result.reason_code == GUARD_CALLER_MISMATCH


class TestBuildRecoveryInboxMessageParams:
    def test_params_carry_the_exact_suffix_and_a_deterministic_key(self):
        callback = _callback()
        analysis = analyze_native_dispatch_corruption(
            "bound partsuffix bytes", [utf8_sha256("bound part")]
        )
        params = build_recovery_inbox_message_params(callback=callback, analysis=analysis)
        assert params["sender_id"] == callback.worker_terminal_id
        assert params["receiver_id"] == callback.caller_id
        assert params["message"] == analysis.unbound_suffix
        assert params["origin"] == InboxMessageOrigin.SYSTEM
        assert params["assignment_id"] == callback.assignment_id
        assert params["idempotency_key"] == (
            f"native-dispatch-recovery:{callback.assignment_id}:{analysis.unbound_suffix_sha256}"
        )

    def test_refuses_to_build_params_with_no_matched_suffix(self):
        callback = _callback()
        exact = "nothing to recover here"
        analysis = analyze_native_dispatch_corruption(exact, [utf8_sha256(exact)])
        with pytest.raises(ValueError, match="matched unbound suffix"):
            build_recovery_inbox_message_params(callback=callback, analysis=analysis)


@pytest.fixture
def recovery_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'native-dispatch-recovery.sqlite'}",
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


class TestApplyRecoveryInboxMessage:
    """apply_recovery_inbox_message is unit tested here against a real,
    synthetic, throwaway SQLite database -- it is never invoked anywhere in
    this codebase against live data (see the module docstring).
    """

    def test_creates_exactly_one_new_durable_pending_message(self, recovery_db):
        db.create_terminal("synthetic-caller", "cao-session", "caller-win", "mock_cli")
        callback = _callback()
        analysis = analyze_native_dispatch_corruption(
            "bound partsuffix bytes", [utf8_sha256("bound part")]
        )
        params = build_recovery_inbox_message_params(callback=callback, analysis=analysis)

        created = apply_recovery_inbox_message(params)

        assert created.message == "suffix bytes"
        assert created.sender_id == callback.worker_terminal_id
        assert created.receiver_id == callback.caller_id
        assert created.origin == InboxMessageOrigin.SYSTEM
        stored = db.get_inbox_messages(callback.caller_id, limit=10)
        assert [row.id for row in stored] == [created.id]

    def test_byte_identical_replay_is_idempotent(self, recovery_db):
        db.create_terminal("synthetic-caller", "cao-session", "caller-win", "mock_cli")
        callback = _callback()
        analysis = analyze_native_dispatch_corruption(
            "bound partsuffix bytes", [utf8_sha256("bound part")]
        )
        params = build_recovery_inbox_message_params(callback=callback, analysis=analysis)

        first = apply_recovery_inbox_message(params)
        second = apply_recovery_inbox_message(params)

        assert first.id == second.id
        stored = db.get_inbox_messages(callback.caller_id, limit=10)
        assert len(stored) == 1

    def test_non_identical_collision_under_the_same_key_is_refused(self, recovery_db):
        db.create_terminal("synthetic-caller", "cao-session", "caller-win", "mock_cli")
        callback = _callback()
        analysis = analyze_native_dispatch_corruption(
            "bound partsuffix bytes", [utf8_sha256("bound part")]
        )
        params = build_recovery_inbox_message_params(callback=callback, analysis=analysis)
        apply_recovery_inbox_message(params)

        tampered = dict(params)
        tampered["message"] = "a different message under the same idempotency key"
        with pytest.raises(ValueError, match="idempotency key collision"):
            apply_recovery_inbox_message(tampered)
