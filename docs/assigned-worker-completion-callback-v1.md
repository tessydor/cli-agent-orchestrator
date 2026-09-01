# Assigned-worker completion callbacks V1

This design makes successful `assign` completion observable to the persisted
caller even when the worker model never invokes `send_message`. It is based on
commit `0b02db7a04af4b7a5519dea383a663a0400517d6`; it does not require a newer CAO
base.

## Contract

An accepted assigned worker owns three immutable routing values created before
task delivery:

- `assignment_id`: identity of the accepted task;
- `completion_id`: identity and idempotency seed of its one success callback;
- `worker_terminal_id` and `caller_id`: the original route, which is never
  inferred again or automatically changed.

The terminal row and callback row commit in one SQLite transaction. The callback
row deliberately has no terminal foreign key, so its final report and audit
history survive deletion of either terminal.

Only a provider-derived `COMPLETED` terminal state after the assigned input was
durably marked `DISPATCHED` can capture success. Text printed by the model is not
a completion signal. `ERROR`, creation failure, and retirement before success
become `FAILED` or `CANCELLED` and never emit a success callback.

## State machine

Task lifecycle and delivery state are independent:

| Task lifecycle | Meaning |
| --- | --- |
| `ASSIGNED` | Route and identities exist, but task acceptance is not proven. |
| `DISPATCHED` | Initial assigned input was accepted. |
| `COMPLETED` | A genuine completed status was observed and final output is durable. |
| `FAILED` / `CANCELLED` | No success callback is permitted. |

| Delivery state | Durable transition |
| --- | --- |
| `NOT_READY` | Assignment exists; no successful final report exists. |
| `CAPTURED` | Full final report, SHA-256, and logical result reference committed. |
| `DELIVERING` | An attempt and its timestamps committed before enqueue. |
| `ENQUEUED` | The callback row links to the committed inbox row. |
| `ACKNOWLEDGED` | The exact inbox row, route, assignment, origin, and idempotency key were re-read and verified. |
| `SUPPRESSED_EXPLICIT` | An equivalent explicit final `send_message` row already satisfies the callback. |
| `RETRYABLE` | Capture, receiver classification, or enqueue can be reconciled. |
| `TERMINAL_ERROR` | The immutable receiver is deleted/permanently invalid, or the task failed/cancelled; report and audit remain readable. |

Acknowledgement means durable visibility in the inbox, not an unobservable claim
that terminal keystrokes have been consumed. The unique inbox key is
`assigned-worker-completion:<completion_id>`. Reusing it returns the original row
only when all immutable payload and route fields match; any collision fails.

## Event and crash ordering

Before `COMPLETED` becomes visible through either the status API or event bus,
StatusMonitor installs an in-memory capture barrier for known assigned workers.
Inbox delivery leaves queued rows `PENDING` while that barrier is closed. The
barrier opens only after the final report commits, preventing a queued next turn
from overwriting the only extractable final response.

At process startup, all unfinished assignments are registered before status and
inbox consumers start. Background reconciliation then resumes these committed
boundaries:

| Interrupted point | Recovery behavior |
| --- | --- |
| Before final capture | Re-read genuine provider status and retry capture. |
| After capture, before inbox insert | Resume from the retained report. |
| After inbox commit, before callback link | Reuse the unique inbox row. |
| After link, before acknowledgement | Verify and acknowledge the same row. |
| After acknowledgement, before immediate terminal wake | Ordinary inbox reconciliation delivers the retained `PENDING` row. |

Duplicate events and concurrent in-process reconciliation serialize per worker.
Successful or explicitly suppressed terminal states are never enqueued again.
Pending assignment callbacks remain retryable after backend paste failures and
are exempt from age cleanup until delivery leaves `PENDING`.

## Receiver and explicit-message behavior

Receiver audit state distinguishes `ACTIVE`, `RETAINED_UNREACHABLE`, `DELETED`,
`RETRYABLE_FAILURE`, and `PERMANENTLY_INVALID`.

- Active and retained-but-unreachable callers receive exactly one durable inbox
  row. The latter remains pending for later recovery.
- Transient lookup/backend failures stay retryable without enqueueing early.
- Deleted and permanently invalid callers become terminal errors. The route is
  never redirected, and the final report remains available for manual recovery.
- Explicit `send_message` traffic is unchanged and is tagged for audit. At real
  completion, only a conservatively equivalent final message for the same
  assignment suppresses the server row. Unrelated progress/intermediate messages
  remain independent.

The manual recovery surface is:

```text
GET /assigned-workers/{worker_terminal_id}/completion-callback
```

It requires read or admin scope when API authentication is enabled and returns
the retained report, hash, immutable route, state, attempts, timestamps, and
terminal error.

## Production update procedure (not executed)

1. Record the approved PR head and back up the CAO SQLite database with its
   restrictive permissions preserved.
2. Drain or explicitly account for in-flight assignments.
3. Install the exact approved fork commit into the existing uv tool environment,
   for example `uv tool install git+https://github.com/tessydor/cli-agent-orchestrator.git@<approved-head> --upgrade --reinstall`.
4. Restart the existing `cao-server` through its current service-management
   mechanism. V1 migrations run at startup; a failed idempotency migration stops
   startup rather than running without the unique callback constraint.
5. Verify health, create an isolated synthetic assigned worker, and inspect its
   callback audit endpoint before returning production traffic.

No production install or restart was performed while implementing V1.

## Rollback

Stop new assignments, preserve the SQLite database and any retained reports, and
reinstall the previous known-good commit with `--reinstall`; then restart through
the same operator-controlled mechanism. The schema changes are additive, so the
older server ignores the callback table and extra inbox columns. Before rollback,
manually recover any captured-but-unacknowledged reports because the older server
does not reconcile them and its ordinary retention job does not know their
pending-delivery exemption.

## V1 limitations

- Exactly-once is defined at the durable callback inbox row. Existing inbox
  delivery marks a row delivered before terminal paste to prevent duplicates, so
  a process crash in that narrow legacy boundary can avoid duplication at the
  cost of requiring manual recovery from the retained report.
- Callback/report rows have no automatic retention policy in V1. This favors
  audit and recovery but requires a future owner-approved pruning policy for
  long-running installations.
- Final extraction remains provider-specific and bounded by the provider's
  existing extraction rules. Extraction errors are retained as retryable and
  block worker retirement rather than discarding the terminal.
- The patch is intentionally based on the installed provenance commit. Later
  `main` has substantial orchestration changes, so forward-port conflicts must be
  resolved deliberately rather than by changing the V1 base.
