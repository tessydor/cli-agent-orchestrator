# Native Claude assignments

New Claude ASSIGN terminals retain their durable assignment/completion IDs but
launch the interactive Claude Code TUI. A private transport marker is persisted
before provider construction. Existing assigned terminals without that marker
keep their legacy streaming transport when CAO reconstructs their provider.

Claude Code 2.1.196 or newer is required for prompt correlation; the live check
used 2.1.263. A native assignment starts a fresh conversation and cannot also
resume a different Claude conversation. Operator-launched, unassigned terminals
retain their existing behavior.

## Result capture

UserPromptSubmit records the native prompt UUID and a SHA-256 digest of the
actual prompt. It must match the exact post-memory-injection dispatch bound by
CAO. Stop/StopFailure can retain a report only for that correlated prompt and
session. Subagent events and unrelated follow-ups cannot replace the original
assignment result. Reports use the existing immutable provider-report validator,
completion delivery, restart reconciliation, and retirement guards.

A completed provider turn is not proof that the task succeeded: a refusal is
retained verbatim and delivered to the supervisor even without send_message.
Brain must review the actual result before accepting work or retiring a worker.
The adapter never reopens a refusal or answers a permission check.

A native provider that returns to a shell reports ERROR, including after
provider reconstruction on server restart. An interrupted turn with no Stop
report cannot be promoted to a successful final result.

## Supervisor tools

- get_terminal_context reads this process's terminal, recorded caller/profile,
  assignment/completion IDs, and working directory through CAO HTTP APIs.
- inspect_worker reads status and bounded output only for a direct worker in the
  caller's session.
- A waiting Claude worker generates an idempotent SYSTEM inbox notice to its
  recorded caller. This does not complete its assignment or answer its question.
- get_worker_question returns a live, numbered native-menu snapshot.
  answer_worker_question accepts a strict index and that snapshot's digest.
  It verifies each navigation step and sends at most one Enter per consumed
  snapshot. Unknown, stale, expired, ambiguous, and free-form menu answers fail
  closed. The old text-answer path is rejected for Claude.

- inspect_terminal_retirement_state reads a worker's durable callback record,
  including a computed `state_token` snapshot -- call this first and pass its
  `assignment_id`/`state_token` unchanged into reconcile_terminal_retirement.
- reconcile_terminal_retirement lets the recorded assigning caller record
  durable, evidence-backed acceptance of a worker whose authoritative provider
  completion report will never become available (for example an old/restarted
  native session), so delete_terminal stops returning 409 for it. It never
  fabricates a completion callback or changes the retained lifecycle/
  delivery_state/final_result: it is a distinct, additive, auditable
  who/when/why/evidence fact that `prepare_terminal_retirement` accepts as
  sufficient, on its own, to allow retirement -- and re-checks liveness again
  at the actual retirement moment, not just at reconciliation time. Refused
  for a wrong caller, a wrong/stale assignment snapshot, a live terminal, one
  waiting on a decision, or a record that already has a genuine provider
  report or ordinary FAILED/CANCELLED disposition. Idempotent; an exact
  replay is a no-op, but a repeat call with different evidence is refused as
  a conflict rather than overwriting the first durable record.

  The underlying REST route additionally requires this server's own local
  machine service token (`require_local_service_token`, `security/auth.py`),
  not merely `cao:admin` scope -- a generic admin-scoped caller (a human
  operator's session, a dashboard, an unrelated service) that reads and
  resubmits the exact recorded `caller_id` is refused before it ever reaches
  the caller_id check, because it does not hold this deployment's shared
  local secret. This still does not bind to a SPECIFIC terminal -- every
  local MCP subprocess shares that one secret -- so the caller_id check
  remains the only defense against one legitimate terminal reconciling a
  different terminal's assignment.

CAO terminal IDs and model-native ListAgents IDs are different namespaces.
These tools expose routing records under CAO's existing local trust/auth model;
they are not cryptographic owner authorization and do not add tenant isolation.

Only delegated routine questions may be answered by Brain. Human identity
checks and owner-reserved approvals remain human decisions. No hook modifies
permission decisions, incident memory, or approval history.

## Boundaries

Menu recognition is deliberately conservative: a fully visible, uniquely
highlighted numbered menu and known footer are required. Nonstandard/localized
widgets may be rejected. The question cache is process-local and invalidated by
server restart; a fresh inspection is required. Concurrent manual interaction
can invalidate a snapshot. Do not retry an uncertain key delivery automatically.

Completion capture assumes the native hooks remain enabled. Missing correlation,
disabled hooks, conflicting Stop reports (for example another hook continuing
the turn), and unsupported CLI versions fail closed. They require inspection;
CAO must not infer success from stale screen text.

A terminal bootstrapped without ASSIGN metadata has no automatic assigned
completion record. This change does not retroactively register existing refused
workers. Inspect and retain them using the existing lifecycle.

## Validation and rollout

The default non-integration suite passed 8,951 tests (31 skipped, 253 deselected,
one expected failure) before the final targeted hardening additions. Targeted
tests cover native launch, callback delivery of refusal, API-error capture,
question notification/deduplication, stale-menu rejection, and restored-shell
detection. The final targeted runs passed 116 lifecycle/question tests and 132 creation/provider
tests. The new ASSIGN regression fails on the merged maintenance base because
the launch command contains --print, and passes with this change.

A live native Claude 2.1.263 smoke test with tools disabled verified:
- ordinary TUI display and a Stop report accepted by the production validator;
- exact delivery of a 32,768-byte multiline Unicode follow-up;
- the follow-up did not overwrite the original assignment report.

The temporary test window was closed after completion. This was a native CLI
transport check, not an end-to-end deployed Brain acceptance test.

Before rollout, retain the installed package and CAO state for rollback. Deploy
the reviewed merge using the established installation path; preserve
KillMode=process and verify health and existing sessions after restart. Refresh
the Brain MCP connection and verify the new tools are visible. Run a harmless
assigned task through Brain, a routine question, result review, and retirement.
Only then resume the queued Equities task. Roll back the package if that canary
fails; preserve new transport/report sidecars for investigation.

Hook contract: https://code.claude.com/docs/en/hooks
