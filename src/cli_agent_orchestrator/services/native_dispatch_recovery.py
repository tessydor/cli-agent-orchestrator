"""Bounded, fail-closed recovery for a corrupted native-input dispatch.

Background (correction-971): before the terminal_input_lock fix in
terminal_service.py, an assigned worker's initial ASSIGN dispatch could have
an unrelated follow-up message physically pasted into its still-open native
transcript before a turn boundary, so the follow-up's bytes end up appended
to the registered dispatch text with no separator. The dispatch-identity hash
bind_completion_dispatch() recorded for that worker therefore covers only a
PREFIX of what the native provider actually received; the remaining suffix
is a genuine follow-up message that was never durably delivered as its own
inbox row.

This module analyzes an ALREADY-corrupted assignment from archived evidence
(a read-only transcript copy, the assignment's own registered dispatch
digests) and, only if every fail-closed guard passes, produces the parameters
for the ONE legitimate durable write recovery may ever make: re-queuing the
proven unbound suffix as its own new, distinct PENDING inbox message via the
existing, already-audited create_inbox_message() plumbing. It never edits
provider/native transcript history, never mutates lifecycle/delivery_state/
final_result, never forges a completion, and never silently marks anything
delivered.

This is a general, reusable capability -- it does not know about, and is not
scoped to, any specific incident. Binding a matched unbound suffix to an
exact durable message record in some OTHER system (so a caller can supply
`durable_suffix_reference` with confidence) is explicitly the caller's
responsibility; when that binding cannot be established, check_recovery_
guards fails closed with GUARD_DURABLE_SUFFIX_UNBOUND rather than guessing.

apply_recovery_inbox_message() performs a real write and is fully unit
tested, but nothing in this codebase calls it against live data -- executing
it against a real corrupted assignment is a separate, explicit, owner-
authorized operator action outside this module's and this PR's scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from cli_agent_orchestrator.clients.database import create_inbox_message
from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignmentLifecycle,
)
from cli_agent_orchestrator.models.inbox import InboxMessage, InboxMessageOrigin
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
GUARD_DURABLE_SUFFIX_UNBOUND = "unbound_suffix_not_durably_recorded"

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
    bound_prefix_length: int | None
    matched_digest: str | None
    unbound_suffix: str | None
    unbound_suffix_sha256: str | None


def analyze_native_dispatch_corruption(
    transcript_first_user_text: str,
    admissible_dispatch_sha256: list[str],
) -> DispatchCorruptionAnalysis:
    """Find the longest prefix of ``transcript_first_user_text`` whose SHA256
    is one of ``admissible_dispatch_sha256``, and split off everything after
    it as the unbound suffix.

    Pure and read-only: no I/O, no mutation, no claim about whether the
    suffix is SAFE to recover -- only where the immutable, already-durable
    dispatch boundary provably ends. Independently recomputed from the raw
    text every time; never trust an externally supplied length or hash claim
    in its place (see check_recovery_guards' evidence-mutation guard for the
    matching discipline this is meant to pair with).
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
class RecoveryGuardResult:
    allowed: bool
    reason_code: str | None
    detail: str


def check_recovery_guards(
    *,
    requesting_caller_id: str,
    callback: AssignedWorkerCallback | None,
    analysis: DispatchCorruptionAnalysis,
    live_status: TerminalStatus,
    archived_transcript_sha256: str,
    expected_archived_transcript_sha256: str,
    durable_suffix_reference: str | None,
) -> RecoveryGuardResult:
    """Every check fails closed: the first failure wins and nothing
    downstream is trusted or evaluated. Passing every guard means it is safe
    to hand ``analysis.unbound_suffix`` to
    ``build_recovery_inbox_message_params`` for a real durable write -- it
    does not perform that write itself.

    Guard order is deliberate: identity and evidence integrity are checked
    before anything derived from the (unauthenticated-until-proven) archived
    transcript is trusted, and liveness is re-checked last, right before any
    caller would act on this result, to minimize the window between the
    check and use.
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
    if not durable_suffix_reference:
        return RecoveryGuardResult(
            False,
            GUARD_DURABLE_SUFFIX_UNBOUND,
            "the unbound suffix cannot be bound to an exact durable message record -- "
            "report this precise blocker rather than guessing at the follow-up's origin",
        )
    return RecoveryGuardResult(True, None, "all guards passed")


def build_recovery_inbox_message_params(
    *,
    callback: AssignedWorkerCallback,
    analysis: DispatchCorruptionAnalysis,
) -> dict[str, Any]:
    """Exact, deterministic parameters for the one legitimate durable write
    this recovery path may ever make: re-queuing the proven unbound suffix as
    its OWN new, distinct inbox message -- never editing native transcript
    history, never touching the original dispatch/callback/final_result.

    Callers MUST have already obtained an ``allowed`` result from
    check_recovery_guards for the same ``callback``/``analysis`` pair; this
    function performs no guard checks of its own; it and
    apply_recovery_inbox_message together are the "safely reviewable
    portion" -- the write itself is always additive-only and idempotent
    (see create_inbox_message's idempotency-key collision handling), never a
    silent delivery-state mutation.
    """
    if not analysis.unbound_suffix or not analysis.unbound_suffix_sha256:
        raise ValueError("build_recovery_inbox_message_params requires a matched unbound suffix")
    idempotency_key = (
        f"native-dispatch-recovery:{callback.assignment_id}:{analysis.unbound_suffix_sha256}"
    )
    return {
        "sender_id": callback.worker_terminal_id,
        "receiver_id": callback.caller_id,
        "message": analysis.unbound_suffix,
        "origin": InboxMessageOrigin.SYSTEM,
        "assignment_id": callback.assignment_id,
        "idempotency_key": idempotency_key,
    }


def apply_recovery_inbox_message(params: dict[str, Any]) -> InboxMessage:
    """Perform the one durable write ``build_recovery_inbox_message_params``
    describes.

    Delegates entirely to the existing, already-audited
    ``create_inbox_message`` -- no parallel persistence path. That function's
    own idempotency-key handling makes a byte-identical replay a safe no-op
    and rejects a non-identical collision under the same key outright, so a
    duplicate or altered recovery attempt for the same assignment/suffix can
    never silently diverge.

    NOT invoked anywhere in this codebase against live data. Exposed and
    unit-tested so an authorized operator with real database access to the
    affected deployment can invoke it explicitly, only after independently
    confirming every check_recovery_guards guard passes there.
    """
    return create_inbox_message(**params)
