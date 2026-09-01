# Assigned-worker completion callbacks V1

This design makes successful `assign` completion observable to the persisted
caller even when the worker model never invokes `send_message`. It is based on
commit `0b02db7a04af4b7a5519dea383a663a0400517d6`; it does not require a newer CAO
base.

## Contract

An assigned worker owns immutable identity and routing values created before
task delivery:

- `assignment_id`: identity of the accepted task;
- `completion_id`: identity and idempotency seed of its one success callback;
- `worker_terminal_id` and `caller_id`: the original route, which is never
  inferred again or automatically changed; and
- `routing_digest`: a deterministic digest over all four identity and routing
  fields.

The terminal row and callback row commit in one SQLite transaction. The callback
row deliberately has no terminal foreign key, so its captured report and audit
history survive deletion of either terminal. SQLite triggers make the route,
captured result (even after an attempted lifecycle downgrade), callback link,
and linked or server-origin inbox route/payload immutable. Every service and
manual-recovery read independently validates the routing digest, result SHA-256
and reference, lifecycle/delivery shape, timestamp ordering, and linked inbox
evidence. A mismatch fails closed without returning report text.

Only a provider-derived `COMPLETED` state after the assigned input was durably
marked `DISPATCHED` can capture success. Text printed by the model is not a
completion signal. In particular, a `PROCESSING` or `COMPLETED` provider state
observed after restart does not prove that an `ASSIGNED` prompt was accepted.
Such a record becomes auditable `UNRESOLVED` / `MANUAL_RECOVERY`, and its
terminal remains the recovery handle. `ERROR`, creation failure, and retirement
before success become `FAILED` or `CANCELLED` and never emit a success callback.

## State machine

Task lifecycle and delivery state are independent:

| Task lifecycle | Meaning |
| --- | --- |
| `ASSIGNED` | Route and identities exist, but task acceptance is not proven. |
| `DISPATCHED` | Initial assigned input was accepted. |
| `COMPLETED` | Genuine completion was observed and final output is durable. |
| `UNRESOLVED` | Dispatch is unproven; retain the terminal for recovery. |
| `FAILED` / `CANCELLED` | No success callback is permitted. |

| Delivery state | Durable transition |
| --- | --- |
| `NOT_READY` | Assignment exists; no successful final report exists. |
| `CAPTURED` | Report, SHA-256, and logical result reference committed. |
| `DELIVERING` | An enqueue attempt and its timestamps committed. |
| `ENQUEUED` | Legacy recovery state linking a committed callback inbox row. |
| `ACKNOWLEDGED` | SQLite atomically accepted and linked the callback row. |
| `SUPPRESSED_EXPLICIT` | An equivalent explicit final satisfies the callback. |
| `RETRYABLE` | Failure is retained for server retry. |
| `MANUAL_RECOVERY` | Automation cannot prove a safe next transition. |
| `TERMINAL_ERROR` | No automatic path remains; retain data for recovery. |

`ACKNOWLEDGED` is an enqueue acknowledgement: the exact route, assignment,
payload, origin, and idempotency identity are committed and linked in SQLite. It
does not claim that terminal paste has happened. The unique server key is
`assigned-worker-completion:<completion_id>`. Reusing it returns the original
row only when all immutable fields match; a collision fails closed.

## Event, enqueue, and crash ordering

Before `COMPLETED` becomes visible through either the status API or event bus,
StatusMonitor installs an in-memory capture barrier for known assigned workers.
Inbox delivery leaves queued rows `PENDING` while that barrier is closed. The
barrier opens only after the final report commits, preventing a queued next turn
from overwriting the only extractable final response.

At process startup, all unfinished assignments are registered before status and
inbox consumers start. Background reconciliation resumes these durable
boundaries:

| Interrupted point | Recovery behavior |
| --- | --- |
| Before dispatch proof | Retain `UNRESOLVED`; never infer dispatch. |
| After persisted dispatch | Retry provider status and final capture. |
| After capture, before inbox insert | Resume from the retained report. |
| Legacy inbox commit, before link | Retain and reuse the unique row. |
| Legacy link, before acknowledgement | Verify and acknowledge that row. |
| After acknowledgement, before wake | Inbox reconciliation sees `PENDING`. |

After startup, capture, receiver-classification, and enqueue failures that enter
`RETRYABLE` wake one completion-service-owned scheduler. It keeps at most one
deadline per worker, uses exponential delays from one second up to a 60-second
cap, and runs all deadlines through one shared task. Duplicate failure signals
coalesce; there is no supervisor poll, callback-table sweep, or busy loop. A
retry that remains transient schedules the next capped delay, while success or a
terminal classification clears its deadline. Server shutdown cancels the shared
task and clears its in-memory schedule; durable `RETRYABLE` rows are recovered by
the mandatory startup reconciliation on the next process start.

Callback equivalence selection, inbox insertion, callback linkage, and enqueue
acknowledgement now share one `BEGIN IMMEDIATE` transaction. The explicit
`send_message` insertion path uses that same serialization. Therefore either an
equivalent explicit final wins and the server row is omitted, or the server row
wins and the later explicit call returns it. Non-equivalent progress messages
remain independent.

Inbox delivery uses an atomic `PENDING` to `DELIVERING` claim with an opaque
owner token. Only the winning claimant may paste, and only that token can resolve
the row to `DELIVERED`, `FAILED`, or back to `PENDING`. A caught pre-paste send
failure resets the callback for retry. A process crash after a paste but before
claim resolution leaves `DELIVERING` as a deliberate manual-recovery boundary;
it is not automatically replayed because CAO cannot know whether the external
terminal side effect occurred. The completion retry scheduler only reconciles
callback state and does not replay such inbox claims.

## Retention and deletion invariant

Every terminal deletion path uses one database invariant. A worker in
`ASSIGNED`, `DISPATCHED`, or `UNRESOLVED` cannot lose its terminal, pane, or log
recovery route until its outcome is safely captured or classified. Cleanup,
session deletion, flow recycling, Herdr ghost reconciliation, same-name cleanup,
and creation rollback all pass through this rule. Bulk terminal deletion is
implemented as invariant-checked per-row deletion.

Only positive proof that the backend worker is already missing permits the
missing-backend path. That path records `FAILED` / `TERMINAL_ERROR` and an audit
reason before provider cleanup or terminal-row deletion. If provider cleanup is
deferred, the terminal row remains, but the worker no longer dangles in an
unclassified state. A failed terminal retirement also defers deletion of its
containing backend session and session environment.

Deferred initialization failure and exhausted input-acceptance retries are a
separate positive-proof path: the assigned task never crossed the dispatch gate.
That path first persists `FAILED` / `TERMINAL_ERROR`, then uses normal terminal
retirement. Its caller notification is composed only after retirement returns;
it claims deletion only on a true result and otherwise says cleanup was deferred
and the terminal/report recovery handle remains. Notification enqueue failure
does not block an otherwise safe teardown. A worker parked on
`WAITING_USER_ANSWER` is not classified or deleted by this path. Likewise, an
exception after `send_input` begins cannot prove whether its external paste
occurred; an unproven callback becomes `UNRESOLVED` / `MANUAL_RECOVERY` and is
retained instead of being falsely classified as never dispatched.

Captured callback rows, linked inbox evidence, and even unlinked server rows at
the legacy insert-before-link crash boundary are exempt from age cleanup.
Uncaptured workers also retain their `<terminal>.log`, `<terminal>.scrollback`,
and `<terminal>.snapshot.json` recovery artifacts. Ordinary and unlinked
explicit messages retain the existing inbox retention policy.

## Receiver behavior

Receiver audit state distinguishes `ACTIVE`, `RETAINED_UNREACHABLE`, `DELETED`,
`RETRYABLE_FAILURE`, and `PERMANENTLY_INVALID`.

- Active and retained-but-unreachable callers get one durable callback inbox
  row. The latter remains pending for later delivery.
- Transient lookup/backend failures remain retryable without enqueueing early.
- Deleted and permanently invalid callers become terminal errors. The route is
  never redirected, and the final report remains available for manual recovery.
- Deleting a caller with a queued or claimed callback atomically marks the inbox
  row `FAILED` and the callback `TERMINAL_ERROR` / `DELETED`.
- If paste was durably marked `DELIVERED` before later caller deletion, that
  historical success is preserved while receiver state records `DELETED` and
  an audit note makes the ordering explicit.
- Explicit `send_message` traffic is unchanged except that a final-equivalent
  message participates in the atomic suppression contract. Unrelated progress
  and intermediate messages are never suppressed.

The manual recovery surface is:

```text
GET /assigned-workers/{worker_terminal_id}/completion-callback
```

It requires read or admin scope when API authentication is enabled and returns
the retained report, hash, immutable route, state, attempts, timestamps, and
terminal error. Integrity failures return a conflict response without report
content.

## Production update procedure (not executed)

1. Record the approved PR head and back up the CAO SQLite database with its
   restrictive permissions preserved.
2. Drain or explicitly account for in-flight assignments.
3. Install the exact approved fork commit into the existing uv tool environment.
   For example:

   ```shell
   repo=https://github.com/tessydor/cli-agent-orchestrator.git
   uv tool install "git+${repo}@<approved-head>" --upgrade --reinstall
   ```

4. Restart the existing `cao-server` through its current service-management
   mechanism. V1 migrations run at startup; an integrity or idempotency migration
   failure stops startup rather than running without the required guards.
5. Verify health, create an isolated synthetic assigned worker, and inspect its
   callback audit endpoint before returning production traffic.

No production install or restart was performed while implementing V1.

## Rollback

Stop new assignments, preserve the SQLite database and retained reports, and
reinstall the previous known-good commit with `--reinstall`; then restart through
the same operator-controlled mechanism. The schema changes are additive, so the
older server ignores the callback table and extra inbox columns. Before rollback,
manually recover captured but unpasted reports because the older server does not
understand callback reconciliation or its retention exemptions.

## V1 limitations

- V1 guarantees one durable callback row per completion identity and one active
  claimant per inbox delivery attempt. It does not promise exactly-once terminal
  side effects across the unobservable crash boundary after paste and before
  claim resolution. That row remains `DELIVERING` for operator recovery rather
  than risking automatic duplicate paste.
- Callback/report/evidence rows have no automatic pruning policy in V1. This
  favors audit and recovery but requires a future owner-approved retention
  policy for long-running installations.
- The routing digest and route/result triggers detect or reject changes to their
  covered immutable evidence. They do not cryptographically authenticate a
  legitimate-looking, self-consistent mutation of mutable lifecycle/delivery
  fields by a privileged database writer; such a raw SQL lifecycle transition
  can be accepted without removing the route/result triggers.
- Final extraction remains provider-specific and bounded by the provider's
  existing extraction rules. Extraction errors are retryable and block worker
  retirement rather than discarding the terminal.
- The patch intentionally remains based on the installed provenance commit.
  Later `main` has substantial orchestration changes, so forward-port conflicts
  must be resolved deliberately rather than by changing the V1 base.
