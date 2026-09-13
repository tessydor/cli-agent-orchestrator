"""Tests for the correction-971/984 corrupted-native-dispatch recovery path.

All transcript/evidence content here is synthetic fixture data invented for
these tests -- nothing from any real assignment, terminal, or archive is
copied into this repository.
"""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import native_dispatch_recovery as recovery_mod
from cli_agent_orchestrator.services.native_dispatch_recovery import (
    GUARD_CALLER_MISMATCH,
    GUARD_CHANGED_CALLBACK_STATE,
    GUARD_CLAIM_UNBOUND,
    GUARD_CONTENT_MISMATCH,
    GUARD_EVIDENCE_MUTATION,
    GUARD_HASH_PREFIX_MISMATCH,
    GUARD_LIVE_TERMINAL_ACTIVITY,
    GUARD_NOT_FOUND,
    GUARD_UNBOUND_SUFFIX_MISSING,
    GUARD_WRONG_DIRECTION,
    MAX_TRANSCRIPT_BYTES,
    ConcatenatedMessageClaim,
    CorruptionRecord,
    analyze_native_dispatch_corruption,
    apply_corruption_record,
    build_corruption_record,
    check_recovery_guards,
    utf8_sha256,
)

RECORDED_AT = "2026-09-13T00:00:00+00:00"


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


def _claim(
    *,
    message_id=99,
    sender_id="synthetic-caller",
    receiver_id="synthetic-worker",
    content="suffix bytes",
    delivery_state="delivered",
    session_id="synthetic-session",
) -> ConcatenatedMessageClaim:
    return ConcatenatedMessageClaim(
        message_id=message_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=content,
        delivery_state=delivery_state,
        session_id=session_id,
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
                "bound partsuffix bytes", [utf8_sha256("bound part")]
            ),
            live_status=TerminalStatus.IDLE,
            archived_transcript_sha256=transcript_sha,
            expected_archived_transcript_sha256=transcript_sha,
            claim=_claim(content="suffix bytes"),
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

    def test_lifecycle_change_after_analysis_is_rechecked(self):
        """The callback passed in must be freshly re-read at guard-check time,
        not a stale snapshot from when analysis was first performed."""
        stale_ok_callback = _callback(lifecycle=AssignmentLifecycle.DISPATCHED)
        fresh_changed_callback = _callback(lifecycle=AssignmentLifecycle.CANCELLED)
        # A caller who re-reads the callback fresh right before checking
        # guards (the required discipline) gets the CHANGED state honored.
        result = check_recovery_guards(**self._base_kwargs(callback=fresh_changed_callback))
        assert result.allowed is False
        assert result.reason_code == GUARD_CHANGED_CALLBACK_STATE
        # Sanity: the stale snapshot alone would have looked eligible --
        # demonstrating why re-reading fresh at guard time is load-bearing.
        stale_result = check_recovery_guards(**self._base_kwargs(callback=stale_ok_callback))
        assert stale_result.allowed is True

    @pytest.mark.parametrize(
        "status", [TerminalStatus.PROCESSING, TerminalStatus.WAITING_USER_ANSWER]
    )
    def test_live_terminal_activity_fails_closed(self, status):
        result = check_recovery_guards(**self._base_kwargs(live_status=status))
        assert result.allowed is False
        assert result.reason_code == GUARD_LIVE_TERMINAL_ACTIVITY

    def test_unbound_claim_fails_closed_rather_than_guessing(self):
        result = check_recovery_guards(**self._base_kwargs(claim=None))
        assert result.allowed is False
        assert result.reason_code == GUARD_CLAIM_UNBOUND

    def test_wrong_content_fails_closed(self):
        result = check_recovery_guards(
            **self._base_kwargs(claim=_claim(content="a completely different message"))
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_CONTENT_MISMATCH

    def test_wrong_message_hash_via_altered_content_fails_closed(self):
        # Same length, different bytes -- proves the check is a real content
        # comparison, not merely a length check.
        result = check_recovery_guards(**self._base_kwargs(claim=_claim(content="suffix BYTES")))
        assert result.allowed is False
        assert result.reason_code == GUARD_CONTENT_MISMATCH

    def test_wrong_sender_fails_closed(self):
        result = check_recovery_guards(
            **self._base_kwargs(claim=_claim(sender_id="not-the-caller"))
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_WRONG_DIRECTION

    def test_wrong_receiver_fails_closed(self):
        result = check_recovery_guards(
            **self._base_kwargs(claim=_claim(receiver_id="not-the-worker"))
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_WRONG_DIRECTION

    def test_reversed_direction_worker_to_caller_fails_closed(self):
        """The exact provenance bug correction-984 found: a claim in the
        WRONG (worker->caller) direction must never pass."""
        result = check_recovery_guards(
            **self._base_kwargs(
                claim=_claim(sender_id="synthetic-worker", receiver_id="synthetic-caller")
            )
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_WRONG_DIRECTION

    def test_wrong_assignment_via_mismatched_callback_fails_closed(self):
        # A claim's sender/receiver correctly matching a DIFFERENT
        # assignment's caller/worker pair must not pass for THIS assignment.
        other_callback = _callback(
            assignment_id="assignment-other",
            worker_terminal_id="other-worker",
            caller_id="other-caller",
        )
        result = check_recovery_guards(
            **self._base_kwargs(
                requesting_caller_id="other-caller",
                callback=other_callback,
                claim=_claim(sender_id="synthetic-caller", receiver_id="synthetic-worker"),
            )
        )
        assert result.allowed is False
        assert result.reason_code == GUARD_WRONG_DIRECTION

    def test_guard_order_identity_and_evidence_checked_before_liveness(self):
        result = check_recovery_guards(
            **self._base_kwargs(
                requesting_caller_id="someone-else",
                expected_archived_transcript_sha256="mismatched",
                live_status=TerminalStatus.PROCESSING,
                claim=None,
            )
        )
        assert result.reason_code == GUARD_CALLER_MISMATCH


class TestBuildCorruptionRecord:
    def test_record_captures_exact_truthful_evidence(self):
        callback = _callback()
        analysis = analyze_native_dispatch_corruption(
            "bound partsuffix bytes", [utf8_sha256("bound part")]
        )
        claim = _claim(content="suffix bytes")
        record = build_corruption_record(
            callback=callback,
            analysis=analysis,
            claim=claim,
            transcript_sha256="a" * 64,
            recorded_at=RECORDED_AT,
        )
        assert record.assignment_id == callback.assignment_id
        assert record.completion_id == callback.completion_id
        assert record.worker_terminal_id == callback.worker_terminal_id
        assert record.caller_id == callback.caller_id
        assert record.registered_dispatch_sha256 == analysis.matched_digest
        assert record.bound_prefix_length == analysis.bound_prefix_length
        assert record.transcript_sha256 == "a" * 64
        assert record.concatenated_message_id == claim.message_id
        assert record.concatenated_message_sender_id == claim.sender_id
        assert record.concatenated_message_receiver_id == claim.receiver_id
        assert record.concatenated_message_content_sha256 == utf8_sha256(claim.content)
        assert record.concatenated_message_delivery_state == claim.delivery_state
        assert record.concatenated_message_session_id == claim.session_id
        assert record.recorded_at == RECORDED_AT
        # Never a completion, never a report: nothing here resembles
        # final_result/lifecycle/delivery_state mutation fields.
        assert not hasattr(record, "final_result")
        assert not hasattr(record, "lifecycle")

    def test_refuses_to_build_without_a_matched_analysis(self):
        callback = _callback()
        analysis = analyze_native_dispatch_corruption("unrelated text", ["0" * 64])
        assert analysis.matched is False
        with pytest.raises(ValueError, match="matched dispatch analysis"):
            build_corruption_record(
                callback=callback,
                analysis=analysis,
                claim=_claim(),
                transcript_sha256="a" * 64,
                recorded_at=RECORDED_AT,
            )


def _record(**overrides) -> CorruptionRecord:
    callback = _callback()
    analysis = analyze_native_dispatch_corruption(
        "bound partsuffix bytes", [utf8_sha256("bound part")]
    )
    claim = _claim(content="suffix bytes")
    record = build_corruption_record(
        callback=callback,
        analysis=analysis,
        claim=claim,
        transcript_sha256="a" * 64,
        recorded_at=RECORDED_AT,
    )
    if overrides:
        record = CorruptionRecord(**{**record.__dict__, **overrides})
    return record


class TestApplyCorruptionRecord:
    """apply_corruption_record never writes to any real database in these
    tests (or anywhere in this codebase) -- ``persist`` is always a fake, and
    the module docstring explains why the real storage target is left to the
    owner to decide.
    """

    def test_first_apply_persists_once(self):
        persisted = []
        record = _record()
        result = apply_corruption_record(record, existing_record=None, persist=persisted.append)
        assert result == record
        assert persisted == [record]

    def test_byte_identical_replay_is_idempotent_and_does_not_persist_again(self):
        persisted = []
        record = _record()
        result = apply_corruption_record(record, existing_record=record, persist=persisted.append)
        assert result == record
        assert persisted == []  # NOT called again -- already durable

    def test_conflicting_record_for_the_same_key_is_refused(self):
        original = _record()
        conflicting = _record(concatenated_message_content_sha256="different" * 8)
        assert original.record_key() == conflicting.record_key()
        with pytest.raises(ValueError, match="collision"):
            apply_corruption_record(conflicting, existing_record=original, persist=lambda r: None)

    def test_no_duplicate_or_fabricated_completion_across_concurrent_apply(self):
        """Concurrent identical applies against ONE shared fake store must
        result in exactly one persisted record, never two, and never any
        field resembling a fabricated final_result/completion."""
        record = _record()
        store_lock = threading.Lock()
        store: dict[str, CorruptionRecord] = {}
        persist_calls = []

        def transactional_apply():
            # Models "read existing under the same lock/transaction the
            # write uses" -- the discipline apply_corruption_record's own
            # docstring requires of callers.
            with store_lock:
                existing = store.get(record.record_key())
                result = apply_corruption_record(
                    record, existing_record=existing, persist=persist_calls.append
                )
                store[record.record_key()] = result
            return result

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(transactional_apply) for _ in range(8)]
            results = [f.result(timeout=5) for f in futures]

        assert all(r == record for r in results)
        assert (
            len(persist_calls) == 1
        ), f"expected exactly one durable persist call, got {len(persist_calls)}"
        assert len(store) == 1


class TestModuleNeverCreatesOrReplaysMessages:
    """Structural guarantee (correction-984): this module must never create,
    replay, or re-deliver any inbox message, and must carry no field
    resembling a fabricated completion/report. A previous version of this
    module did exactly that (worker->caller SYSTEM inbox message,
    re-queuing the concatenated follow-up) -- this is an enduring regression
    guard, not merely true today.
    """

    def test_module_never_imports_inbox_message_creation(self):
        import ast

        tree = ast.parse(inspect.getsource(recovery_mod))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
        forbidden = {"create_inbox_message", "InboxMessageOrigin", "resolve_inbox_claim"}
        assert not (imported_names & forbidden), (
            "native_dispatch_recovery.py must never import inbox-message creation/"
            f"delivery machinery -- found: {imported_names & forbidden}"
        )

    def test_corruption_record_has_no_completion_or_report_shaped_fields(self):
        record = _record()
        forbidden_fields = {
            "final_result",
            "final_result_sha256",
            "lifecycle",
            "delivery_state",
            "receiver_state",
            "completion",
        }
        actual_fields = set(record.__dataclass_fields__)
        assert not (actual_fields & forbidden_fields), (
            f"CorruptionRecord must never carry completion/report-shaped fields, "
            f"found: {actual_fields & forbidden_fields}"
        )
