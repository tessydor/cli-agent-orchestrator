"""Bounded, fail-closed recovery for a corrupted native-input dispatch.

Background (correction-971): before the terminal_input_lock fix in
terminal_service.py, an assigned worker's initial ASSIGN dispatch could have
an unrelated follow-up message physically pasted into its still-open native
transcript before a turn boundary, so the follow-up's bytes end up appended
to the registered dispatch text with no separator. The dispatch-identity hash
bind_completion_dispatch() recorded for that worker therefore covers only a
PREFIX of what the native provider actually received; the remaining suffix
is a genuine follow-up message.

IMPORTANT provenance correction (correction-984): the concatenated suffix is
a message the ASSIGNING CALLER sent TO the worker (the exact same direction
as the original dispatch itself -- both are native INPUT the worker's
process was supposed to receive as two distinct turns). It is proof that a
prior caller->worker follow-up got corrupted into the dispatch transcript --
it is NEVER a worker-produced report, and recovering it is never equivalent
to "the missing assignment result arrived." An earlier version of this
module got this backwards: it built parameters for a NEW inbox message
FROM the worker TO the caller (worker->caller, origin=SYSTEM), which would
have fabricated a worker report out of what is actually caller input, and
silently released the concatenated message. That behavior has been removed
entirely -- this module now NEVER creates, replays, or re-delivers any inbox
message. It also never claims to restore a missing assignment result and
never releases any OTHER pending message's delivery barrier; only the
proper, already-existing lifecycle transitions may do that.

What this module DOES do: pure, read-only, no-I/O analysis and guard logic --
analyze_native_dispatch_corruption() matches an already-corrupted assignment's
archived transcript against its own registered dispatch digests;
check_recovery_guards() fails closed unless a caller-supplied claim
identifying the durable message believed to have been concatenated in is in
the correct caller->worker direction and its content hash matches the
independently recomputed unbound suffix exactly (plus identity/liveness/
lifecycle/tamper checks); build_corruption_record() assembles a truthful,
nonduplicating CorruptionRecord from a callback+analysis+claim that already
passed those guards. None of this mutates lifecycle/delivery_state/
final_result, forges a completion, or marks anything delivered.

Correction-994 (real DB-backed CAS mutation): the actual durable write --
freshly re-reading the assignment/callback row, rerunning every guard above
inside one transaction, and inserting the resulting CorruptionRecord under
enforced uniqueness -- is clients.database.record_native_dispatch_corruption(),
NOT this module (this module has no database access at all, deliberately;
see test_module_never_imports_inbox_message_creation and its sibling
structural tests). It stores into the dedicated
native_dispatch_corruption_records table (NativeDispatchCorruptionModel),
never assigned_worker_callbacks.reconciliation_evidence (PR #11) -- reusing
that field would conflate this with "caller accepted the completion result,"
which a dispatch-corruption record is not, and would additionally make the
assignment eligible for retirement as a side effect nobody asked for here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignmentLifecycle,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus

# Mirrors provider_completion_report.MAX_REPORT_BYTES: this is an offline
# forensic tool, not a hot path, but analysis is O(n^2) in the transcript
# length (see analyze_native_dispatch_corruption), so a hard upper bound is
# load-bearing, not cosmetic.
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024

# Fail-closed guard reason codes returned by check_recovery_guards.
GUARD_NOT_FOUND = "not_found"
GUARD_CALLER_MISMATCH = "caller_mismatch"
GUARD_EVIDENCE_MUTATION = "evidence_mutation"
GUARD_HASH_PREFIX_MISMATCH = "hash_prefix_mismatch"
GUARD_UNBOUND_SUFFIX_MISSING = "unbound_suffix_missing"
GUARD_CHANGED_CALLBACK_STATE = "changed_callback_state"
GUARD_LIVE_TERMINAL_ACTIVITY = "live_terminal_activity"
GUARD_CLAIM_UNBOUND = "concatenated_message_claim_unbound"
GUARD_CONTENT_MISMATCH = "concatenated_message_content_mismatch"
GUARD_WRONG_DIRECTION = "concatenated_message_wrong_direction"

_ELIGIBLE_LIFECYCLES = (
    AssignmentLifecycle.DISPATCHED,
    AssignmentLifecycle.COMPLETED,
    AssignmentLifecycle.UNRESOLVED,
)


def utf8_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DispatchCorruptionAnalysis:
    """Pure result of matching a transcript's first-user text against the
    assignment's own admissible bound-dispatch digests.

    ``matched`` is False only when NO admissible digest matches any prefix at
    all (including the empty prefix never matching) -- that is itself a
    fail-closed signal, not evidence of anything recoverable.
    """

    matched: bool
    bound_prefix_length: Optional[int]
    matched_digest: Optional[str]
    unbound_suffix: Optional[str]
    unbound_suffix_sha256: Optional[str]


def analyze_native_dispatch_corruption(
    transcript_first_user_text: str,
    admissible_dispatch_sha256: list[str],
) -> DispatchCorruptionAnalysis:
    """Find the longest prefix of ``transcript_first_user_text`` whose SHA256
    is one of ``admissible_dispatch_sha256``, and split off everything after
    it as the unbound suffix.

    Pure and read-only: no I/O, no mutation, no claim about whether the
    suffix is SAFE to recover, and no claim about what the suffix IS (see
    ConcatenatedMessageClaim for that) -- only where the immutable,
    already-durable dispatch boundary provably ends. Independently
    recomputed from the raw text every time; never trust an externally
    supplied length or hash claim in its place.
    """
    encoded_len = len(transcript_first_user_text.encode("utf-8"))
    if encoded_len > MAX_TRANSCRIPT_BYTES:
        raise ValueError(
            f"transcript ({encoded_len} bytes) exceeds the {MAX_TRANSCRIPT_BYTES}-byte "
            "limit for recovery analysis"
        )
    admissible = set(admissible_dispatch_sha256)
    text = transcript_first_user_text
    for length in range(len(text), 0, -1):
        prefix = text[:length]
        digest = utf8_sha256(prefix)
        if digest in admissible:
            suffix = text[length:]
            return DispatchCorruptionAnalysis(
                matched=True,
                bound_prefix_length=length,
                matched_digest=digest,
                unbound_suffix=suffix or None,
                unbound_suffix_sha256=utf8_sha256(suffix) if suffix else None,
            )
    return DispatchCorruptionAnalysis(
        matched=False,
        bound_prefix_length=None,
        matched_digest=None,
        unbound_suffix=None,
        unbound_suffix_sha256=None,
    )


@dataclass(frozen=True)
class ConcatenatedMessageClaim:
    """A caller's claim about which durable message was concatenated into
    the unbound suffix -- e.g. an operator's independently-verified finding
    from a database this module has no access to.

    Never trusted on its own: check_recovery_guards cross-checks ``content``
    against the independently recomputed ``analysis.unbound_suffix`` byte for
    byte, and checks ``sender_id``/``receiver_id`` are in the caller->worker
    direction the corruption mechanism requires (never worker->caller -- see
    the module docstring's provenance correction). ``message_id`` is an
    opaque identifier in whatever system holds the durable record (an int,
    a string, a composite key) -- this module makes no assumption about its
    shape.
    """

    message_id: Any
    sender_id: str
    receiver_id: str
    content: str
    delivery_state: str
    session_id: Optional[str] = None


@dataclass(frozen=True)
class RecoveryGuardResult:
    allowed: bool
    reason_code: Optional[str]
    detail: str


def check_recovery_guards(
    *,
    requesting_caller_id: str,
    callback: Optional[AssignedWorkerCallback],
    analysis: DispatchCorruptionAnalysis,
    live_status: TerminalStatus,
    archived_transcript_sha256: str,
    expected_archived_transcript_sha256: str,
    claim: Optional[ConcatenatedMessageClaim],
) -> RecoveryGuardResult:
    """Every check fails closed: the first failure wins and nothing
    downstream is trusted or evaluated. Passing every guard means it is safe
    to hand ``analysis``/``claim`` to ``build_corruption_record`` -- it does
    not build or persist anything itself.

    Guard order is deliberate: identity and evidence integrity are checked
    before anything derived from the (unauthenticated-until-proven) archived
    transcript or claim is trusted, and liveness is checked last, right
    before any caller would act on this result, to minimize the window
    between the check and use.
    """
    if callback is None:
        return RecoveryGuardResult(
            False, GUARD_NOT_FOUND, "no assigned-worker callback record for this worker terminal"
        )
    if callback.caller_id != requesting_caller_id:
        return RecoveryGuardResult(
            False,
            GUARD_CALLER_MISMATCH,
            "requesting caller is not this assignment's immutable recorded assigning caller",
        )
    if archived_transcript_sha256 != expected_archived_transcript_sha256:
        return RecoveryGuardResult(
            False,
            GUARD_EVIDENCE_MUTATION,
            "archived transcript content no longer matches its recorded digest",
        )
    if not analysis.matched:
        return RecoveryGuardResult(
            False,
            GUARD_HASH_PREFIX_MISMATCH,
            "no admissible dispatch digest matches any prefix of the archived transcript",
        )
    if not analysis.unbound_suffix:
        return RecoveryGuardResult(
            False,
            GUARD_UNBOUND_SUFFIX_MISSING,
            "the transcript matches a registered dispatch exactly; there is nothing to recover",
        )
    if callback.lifecycle not in _ELIGIBLE_LIFECYCLES:
        return RecoveryGuardResult(
            False,
            GUARD_CHANGED_CALLBACK_STATE,
            f"callback lifecycle is now {callback.lifecycle.value!r}; no longer eligible "
            "for this recovery path",
        )
    if live_status in (TerminalStatus.PROCESSING, TerminalStatus.WAITING_USER_ANSWER):
        return RecoveryGuardResult(
            False,
            GUARD_LIVE_TERMINAL_ACTIVITY,
            f"terminal is currently {live_status.value}; recovery must wait for it to go idle",
        )
    if claim is None:
        return RecoveryGuardResult(
            False,
            GUARD_CLAIM_UNBOUND,
            "the unbound suffix cannot be bound to an exact durable message record -- "
            "report this precise blocker rather than guessing at the follow-up's origin",
        )
    if claim.content != analysis.unbound_suffix:
        return RecoveryGuardResult(
            False,
            GUARD_CONTENT_MISMATCH,
            "the claimed message's content does not byte-for-byte match the independently "
            "recomputed unbound suffix",
        )
    if claim.sender_id != callback.caller_id or claim.receiver_id != callback.worker_terminal_id:
        return RecoveryGuardResult(
            False,
            GUARD_WRONG_DIRECTION,
            "the claimed message is not in the caller->worker direction this corruption "
            "mechanism requires (never worker->caller)",
        )
    return RecoveryGuardResult(True, None, "all guards passed")


@dataclass(frozen=True)
class CorruptionRecord:
    """Truthful, nonduplicating archive of one native-dispatch corruption
    incident. Never a completion, never a report, never itself a release of
    any pending message's delivery barrier -- purely an audit record of what
    was independently proven.
    """

    assignment_id: str
    completion_id: str
    worker_terminal_id: str
    caller_id: str
    registered_dispatch_sha256: str
    bound_prefix_length: int
    transcript_sha256: str
    concatenated_message_id: Any
    concatenated_message_sender_id: str
    concatenated_message_receiver_id: str
    concatenated_message_content_sha256: str
    concatenated_message_delivery_state: str
    concatenated_message_session_id: Optional[str]
    recorded_at: str

    def record_key(self) -> str:
        """Deterministic identity for idempotency/conflict checks -- one
        corruption record per (assignment, exact concatenated message).
        """
        return f"native-dispatch-corruption:{self.assignment_id}:{self.concatenated_message_id}"


def build_corruption_record(
    *,
    callback: AssignedWorkerCallback,
    analysis: DispatchCorruptionAnalysis,
    claim: ConcatenatedMessageClaim,
    transcript_sha256: str,
    recorded_at: str,
) -> CorruptionRecord:
    """Build the record. Callers MUST have already obtained an ``allowed``
    result from check_recovery_guards for the same arguments -- this
    function performs no guard checks of its own.
    """
    if not analysis.matched or not analysis.matched_digest or analysis.bound_prefix_length is None:
        raise ValueError("build_corruption_record requires a matched dispatch analysis")
    return CorruptionRecord(
        assignment_id=callback.assignment_id,
        completion_id=callback.completion_id,
        worker_terminal_id=callback.worker_terminal_id,
        caller_id=callback.caller_id,
        registered_dispatch_sha256=analysis.matched_digest,
        bound_prefix_length=analysis.bound_prefix_length,
        transcript_sha256=transcript_sha256,
        concatenated_message_id=claim.message_id,
        concatenated_message_sender_id=claim.sender_id,
        concatenated_message_receiver_id=claim.receiver_id,
        concatenated_message_content_sha256=utf8_sha256(claim.content),
        concatenated_message_delivery_state=claim.delivery_state,
        concatenated_message_session_id=claim.session_id,
        recorded_at=recorded_at,
    )
