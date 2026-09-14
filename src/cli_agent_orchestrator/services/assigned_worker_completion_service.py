"""Durable, model-independent assigned-worker completion callbacks.

The service consumes genuine terminal status transitions.  It never infers task
completion from arbitrary model text and never relies on the worker invoking
``send_message``.  Final output is captured before delivery, then driven through
an idempotent inbox enqueue/link/ack state machine that a startup reconciliation
can safely resume after any committed phase.
"""

import asyncio
import logging
import re
import threading
import time
from typing import Any, Optional, Tuple

from sqlalchemy.exc import OperationalError

from cli_agent_orchestrator.backends import TerminalBackendError, TerminalNotFoundError
from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.clients.database import (
    acknowledge_completion_enqueued,
    capture_assigned_worker_completion,
    create_inbox_message,
    get_assigned_worker_callback,
    get_terminal_metadata,
    list_protected_assigned_worker_callbacks,
    list_reconcilable_assigned_worker_callbacks,
    list_retained_assigned_worker_callbacks,
    mark_assigned_worker_dispatched,
    mark_assignment_manual_recovery,
    mark_completion_retryable,
    mark_completion_terminal_error,
    record_completion_delivery_attempt,
)
from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
    format_server_completion_message,
)
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin
from cli_agent_orchestrator.models.provider_completion import (
    ProviderCompletionInvalidError,
    ProviderCompletionReport,
    ProviderCompletionTerminalOutcomeError,
    ProviderCompletionUnavailableError,
    utf8_sha256,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.plugins import PluginRegistry
from cli_agent_orchestrator.providers.manager import provider_manager
from cli_agent_orchestrator.services.event_bus import bus
from cli_agent_orchestrator.services.status_monitor import status_monitor
from cli_agent_orchestrator.utils.event import terminal_id_from_topic

logger = logging.getLogger(__name__)

_TERMINAL_ID_RE = re.compile(r"^[a-f0-9]{8}$")
_DELIVERY_TERMINAL_STATES = frozenset(
    {
        CompletionDeliveryState.ACKNOWLEDGED,
        CompletionDeliveryState.SUPPRESSED_EXPLICIT,
        CompletionDeliveryState.MANUAL_RECOVERY,
        CompletionDeliveryState.TERMINAL_ERROR,
    }
)


class _CaptureBarrier:
    """A capture-release wait/notify primitive built on ONE specific
    worker terminal's own ``terminal_input_lock`` (correction-1007).

    Both existing callers of ``wait_for_capture_before_input``
    (``terminal_service.send_input`` and ``InboxService.deliver_pending``)
    already hold that exact lock for the whole duration of their call --
    that is the entire reason a prior plain ``threading.Event`` was unsafe
    here: blocking on a bare Event while still holding the lock starves
    ANY other thread that needs the SAME lock to perform the release (the
    guarded capture-release transition, correction-997/1001, needs exactly
    this lock for its own fresh-status-read-then-CAS). A
    ``threading.Condition`` built on that SAME lock is the correct,
    standard fix: ``Condition.wait()`` atomically releases the underlying
    lock while blocked and reacquires it before returning, exactly as if
    the waiting thread had briefly exited its own ``with
    terminal_input_lock(...):`` block -- so the release-side transition
    can actually acquire the lock, act, and notify, and the waiter picks
    back up exactly where a normal (non-reentrant) reacquire would leave
    it. No caller-visible behavior changes: from every existing caller's
    point of view this is still "acquire the lock, then block until
    release or timeout."
    """

    __slots__ = ("condition", "released")

    def __init__(self, lock: threading.RLock) -> None:
        self.condition = threading.Condition(lock)
        self.released = False


class AssignedWorkerCompletionService:
    """Capture and deliver successful assigned-worker terminal completions."""

    def __init__(
        self,
        *,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if retry_initial_delay <= 0:
            raise ValueError("retry_initial_delay must be positive")
        if retry_max_delay < retry_initial_delay:
            raise ValueError("retry_max_delay must be at least retry_initial_delay")
        self._registry: Optional[PluginRegistry] = None
        self._worker_locks: dict[str, threading.RLock] = {}
        self._worker_locks_guard = threading.Lock()
        self._known_workers: set[str] = set()
        self._failure_notice_pending: set[str] = set()
        self._capture_barriers: dict[str, _CaptureBarrier] = {}
        self._retry_initial_delay = retry_initial_delay
        self._retry_max_delay = retry_max_delay
        self._retry_lifecycle_guard = threading.Lock()
        self._retry_loop: Optional[asyncio.AbstractEventLoop] = None
        self._retry_wakeup: Optional[asyncio.Event] = None
        self._retry_scheduler_task: Optional[asyncio.Task[None]] = None
        # These maps are accessed only on ``_retry_loop``. One shared scheduler
        # owns every deadline, avoiding one sleeping task per failed callback.
        self._retry_due: dict[str, float] = {}
        self._retry_delays: dict[str, float] = {}

    def _worker_lock(self, worker_terminal_id: str) -> threading.RLock:
        """Return a stable per-worker lock for status/delete/reconcile races."""
        with self._worker_locks_guard:
            return self._worker_locks.setdefault(worker_terminal_id, threading.RLock())

    def register_assignment(self, worker_terminal_id: str) -> None:
        """Register a newly persisted assignment for pre-publish race protection."""
        with self._worker_locks_guard:
            self._known_workers.add(worker_terminal_id)

    def announce_terminal_status(self, worker_terminal_id: str, status: TerminalStatus) -> None:
        """Install a capture barrier before a COMPLETED event is published.

        StatusMonitor calls this synchronously before EventBus publication.  The
        regular InboxService can therefore never win the scheduling race and
        paste a queued next turn over an as-yet-uncaptured final response.
        """
        if status != TerminalStatus.COMPLETED:
            return
        with self._worker_locks_guard:
            if worker_terminal_id not in self._known_workers:
                return
            if worker_terminal_id not in self._capture_barriers:
                # Local import: avoids a module-level import cycle (see
                # reconcile_corrupted_dispatch_capture_release's own local
                # import of the same name, same rationale).
                from cli_agent_orchestrator.services.terminal_service import (
                    terminal_input_lock,
                )

                self._capture_barriers[worker_terminal_id] = _CaptureBarrier(
                    terminal_input_lock(worker_terminal_id)
                )

    def wait_for_capture_before_input(self, worker_terminal_id: str, timeout: float = 5.0) -> bool:
        """Return only when a known completed worker's report is safely durable.

        Unknown/non-assigned terminals have no barrier and retain the existing
        zero-wait inbox path.  On timeout the caller leaves the inbox row PENDING;
        reconciliation retries after capture rather than risking transcript loss.

        Both real callers (``terminal_service.send_input``,
        ``InboxService.deliver_pending``) already hold
        ``terminal_input_lock(worker_terminal_id)`` for their entire call
        into this method -- the barrier's own condition is built on that
        SAME lock (see ``_CaptureBarrier``), so blocking here via
        ``condition.wait()`` correctly releases it for the duration of the
        wait (correction-1007) rather than starving the one path that can
        actually release it (``reconcile_corrupted_dispatch_capture_
        release``, which needs this identical lock for its own guarded
        read-then-CAS).
        """
        with self._worker_locks_guard:
            barrier = self._capture_barriers.get(worker_terminal_id)
        if barrier is None:
            return True
        deadline = time.monotonic() + timeout
        with barrier.condition:
            while not barrier.released:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                barrier.condition.wait(timeout=remaining)
        return True

    def _release_capture_barrier(self, worker_terminal_id: str) -> None:
        with self._worker_locks_guard:
            barrier = self._capture_barriers.pop(worker_terminal_id, None)
            self._known_workers.discard(worker_terminal_id)
        if barrier is not None:
            # notify_all() requires holding the condition's own lock
            # (terminal_input_lock(worker_terminal_id)). Callers that
            # already hold it (reconcile_corrupted_dispatch_capture_
            # release) reenter for free (RLock); callers that don't
            # (the async status-event consumer's normal completion path)
            # acquire it here -- safe, since no path anywhere acquires
            # _worker_lock and terminal_input_lock for the SAME
            # terminal_id in the opposite order (verified: the only two
            # wait_for_capture_before_input callers never touch
            # _worker_lock at all), so this can never invert against
            # anything that holds _worker_lock and waits on this lock.
            with barrier.condition:
                barrier.released = True
                barrier.condition.notify_all()

    def reconcile_corrupted_dispatch_capture_release(
        self,
        worker_terminal_id: str,
        *,
        assignment_id: str,
        requesting_caller_id: str,
        transcript_first_user_text: str,
        expected_transcript_sha256: str,
        admissible_dispatch_sha256: list[str],
        concatenated_message_id: str,
        concatenated_message_sender_id: str,
        concatenated_message_receiver_id: str,
        concatenated_message_content: str,
        concatenated_message_delivery_state: str,
        concatenated_message_session_id: Optional[str],
        recorded_at: str,
        acknowledgement: str,
        acknowledged_at: str,
    ) -> Any:
        """The one supported guarded recovery transition (correction-997/1001).

        Consumes an immutable corrupted-dispatch archive plus an explicit,
        non-empty caller acknowledgement, re-verifies every mutation-time
        binding fresh (via :func:`clients.database.acknowledge_native_dispatch_
        corruption`), records the incident as reconciled/abandoned-for-
        input-capture WITHOUT marking the task successful, and releases ONLY
        this exact dispatch's capture barrier. It never sends or delivers
        anything -- the existing pending follow-up row is left completely
        untouched for ordinary InboxService delivery to observe exactly once.

        Message-1001's TOCTOU finding: a ``live_status`` read taken BEFORE
        acquiring the per-terminal input/capture lock can be stale by the
        time the DB CAS runs -- a concurrent ``send_input``/status detection
        could change what "live" means in between. This method closes that
        gap by acquiring ``terminal_input_lock`` (the same per-terminal
        physical-write/acceptance-fence guard every other native-input entry
        point serializes against -- see terminal_service.terminal_input_lock)
        and reading live status only AFTER acquiring it, immediately before
        the DB call. ``self._worker_lock`` is held OUTER (this class's own
        existing convention for every state-mutating method) purely to
        serialize this transition's callback-row view against this class's
        OTHER worker-lifecycle methods for the same worker; every existing
        caller of ``_worker_lock`` that also touches a lock for THIS SAME
        terminal_id only ever does so via ``wait_for_capture_before_input``
        (which never touches ``_worker_lock`` at all -- see below), so this
        nesting introduces no lock-order inversion between these two locks.

        Message-1007's real-call-graph finding: the claimed ``_worker_lock``
        inversion above did not actually exist in the code (verified by
        reading ``wait_for_capture_before_input``'s body, which only ever
        touches the short-lived ``_worker_locks_guard``) -- but a REAL
        starvation issue did, in ``terminal_input_lock`` alone. Both
        ``send_input`` and ``InboxService.deliver_pending`` hold
        ``terminal_input_lock(worker_terminal_id)`` for their entire call
        into ``wait_for_capture_before_input``; that call used to block on a
        bare ``threading.Event`` while still holding the lock, so a
        concurrent call to THIS method (which needs that same lock to
        perform its guarded read-then-CAS) could never acquire it until the
        waiter's own timeout expired -- functionally a deadlock, bounded
        only by an unrelated caller's timeout constant rather than by
        design. Fixed at the barrier's own definition (see
        ``_CaptureBarrier``): its wait/notify is now a ``threading.
        Condition`` built on that SAME ``terminal_input_lock``, whose
        ``wait()`` atomically releases the lock while blocked and
        reacquires it before returning -- so this method's acquisition
        below can succeed immediately regardless of any concurrent waiter,
        and the waiter resumes normally once notified. This required no
        change to ``send_input``/``deliver_pending``'s own code or lock
        order at all (the "already-live delivery order" is preserved
        exactly); only the barrier's internal wait mechanism changed.

        The barrier release is idempotent regardless of ``released_now``
        (correction-1007 item 2): a process that dies after the DB
        acknowledgement above durably commits but before
        ``_release_capture_barrier`` runs would otherwise strand the
        barrier forever, because a later retry of the SAME acknowledgement
        sees ``released_now=False`` (the DB already reflects this exact,
        matching acknowledgement) and -- under the OLD gated logic -- would
        never attempt release again. ``released_now`` is therefore now only
        an audit/transition signal in the returned record, never a gate on
        whether to release. This is safe to call unconditionally because
        (a) ``_release_capture_barrier`` is already a no-op when no barrier
        is armed for this worker_terminal_id, and (b) reaching this line at
        all already required ``acknowledge_native_dispatch_corruption``'s
        fresh guard revalidation (the exact same guards as message 997) to
        succeed against the CURRENT callback row for this exact assignment
        -- so a stale/reused-terminal_id retry for a since-superseded
        assignment is refused with an exception well before this point,
        never reaching (and therefore never able to mis-release) a
        different, later, unrelated barrier for a reused worker_terminal_id.
        Never happens across any plugin hook or inbox delivery (none occur
        in this path at all, unlike ``_drive_delivery``).

        Raises the same ``NativeDispatchCorruptionGuardError``/``ValueError``
        as ``acknowledge_native_dispatch_corruption`` on any guard failure or
        collision -- this method adds no additional silent-success paths.
        """
        from cli_agent_orchestrator.clients.database import (
            acknowledge_native_dispatch_corruption,
        )
        from cli_agent_orchestrator.services.terminal_service import terminal_input_lock

        with self._worker_lock(worker_terminal_id):
            with terminal_input_lock(worker_terminal_id):
                # Freshly read live status only now, INSIDE the same guard
                # that serializes every physical native-input write and the
                # acceptance fence for this terminal -- never a snapshot
                # read before this lock was acquired.
                live_status = status_monitor.get_status(worker_terminal_id)

                result = acknowledge_native_dispatch_corruption(
                    assignment_id=assignment_id,
                    requesting_caller_id=requesting_caller_id,
                    live_status=live_status,
                    transcript_first_user_text=transcript_first_user_text,
                    expected_transcript_sha256=expected_transcript_sha256,
                    admissible_dispatch_sha256=admissible_dispatch_sha256,
                    concatenated_message_id=concatenated_message_id,
                    concatenated_message_sender_id=concatenated_message_sender_id,
                    concatenated_message_receiver_id=concatenated_message_receiver_id,
                    concatenated_message_content=concatenated_message_content,
                    concatenated_message_delivery_state=concatenated_message_delivery_state,
                    concatenated_message_session_id=concatenated_message_session_id,
                    recorded_at=recorded_at,
                    acknowledgement=acknowledgement,
                    acknowledged_at=acknowledged_at,
                )
                # Fault-injection seam (correction-1007 item 2): a test can
                # monkeypatch this to raise, simulating a process crash
                # exactly at the post-commit/pre-release boundary. A
                # subsequent retry of the identical acknowledgement must
                # still release the barrier below, since the release is
                # unconditional rather than gated on released_now.
                self._capture_release_checkpoint(worker_terminal_id, result)
                self._release_capture_barrier(worker_terminal_id)
                return result

    @staticmethod
    def _capture_release_checkpoint(worker_terminal_id: str, result: Any) -> None:
        """No-op fault-injection seam between the durable acknowledgement
        commit and the in-memory capture-barrier release (correction-1007
        item 2). Mirrors ``_phase_checkpoint``'s existing role for the
        completion-delivery state machine.
        """
        del worker_terminal_id, result

    def start_retry_scheduler(self) -> None:
        """Start the single event-woken retry scheduler on the running loop.

        Production calls this through :meth:`run` before startup reconciliation,
        so a transient failure in that first sweep can always schedule its next
        attempt. Repeated starts on the same loop are idempotent.
        """
        loop = asyncio.get_running_loop()
        with self._retry_lifecycle_guard:
            existing = self._retry_scheduler_task
            if existing is not None and not existing.done():
                if self._retry_loop is not loop:
                    raise RuntimeError("assigned-worker retry scheduler belongs to another loop")
                return
            self._retry_loop = loop
            self._retry_wakeup = asyncio.Event()
            self._retry_due.clear()
            self._retry_delays.clear()
            self._retry_scheduler_task = loop.create_task(self._retry_scheduler())

    async def stop_retry_scheduler(self) -> None:
        """Cancel retry work and release all in-memory deadlines at shutdown."""
        loop = asyncio.get_running_loop()
        with self._retry_lifecycle_guard:
            if self._retry_loop is not None and self._retry_loop is not loop:
                raise RuntimeError("cannot stop assigned-worker retry scheduler from another loop")
            task = self._retry_scheduler_task
            # Clear the published loop first. Threads finishing reconciliation
            # after shutdown will then decline to enqueue more work.
            self._retry_loop = None
            self._retry_wakeup = None
            self._retry_scheduler_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._retry_due.clear()
        self._retry_delays.clear()

    def _request_retry(self, worker_terminal_id: str) -> None:
        """Thread-safely coalesce a delayed retry for one durable callback."""
        with self._retry_lifecycle_guard:
            loop = self._retry_loop
            scheduler = self._retry_scheduler_task
        if loop is None or scheduler is None or scheduler.done() or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._schedule_retry_on_loop, worker_terminal_id, loop)
        except RuntimeError:
            # The loop closed between the guarded read and call_soon_threadsafe.
            # Durable RETRYABLE state remains for the next startup reconciliation.
            logger.debug("Assigned-worker retry loop closed during scheduling")

    def _schedule_retry_on_loop(
        self,
        worker_terminal_id: str,
        expected_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Install one deadline; duplicate failure signals share that deadline."""
        with self._retry_lifecycle_guard:
            if (
                self._retry_loop is not expected_loop
                or self._retry_scheduler_task is None
                or self._retry_scheduler_task.done()
            ):
                return
            wakeup = self._retry_wakeup
        if worker_terminal_id in self._retry_due:
            return
        previous_delay = self._retry_delays.get(worker_terminal_id)
        delay = (
            self._retry_initial_delay
            if previous_delay is None
            else min(self._retry_max_delay, previous_delay * 2)
        )
        self._retry_delays[worker_terminal_id] = delay
        self._retry_due[worker_terminal_id] = expected_loop.time() + delay
        if wakeup is not None:
            wakeup.set()

    def _clear_retry_on_loop(
        self,
        worker_terminal_id: str,
        expected_loop: asyncio.AbstractEventLoop,
    ) -> None:
        with self._retry_lifecycle_guard:
            if self._retry_loop is not expected_loop:
                return
            wakeup = self._retry_wakeup
        self._retry_due.pop(worker_terminal_id, None)
        self._retry_delays.pop(worker_terminal_id, None)
        if wakeup is not None:
            wakeup.set()

    def _clear_requested_retry(self, worker_terminal_id: str) -> None:
        """Drop a stale deadline once a callback reaches a non-retryable state."""
        with self._retry_lifecycle_guard:
            loop = self._retry_loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._clear_retry_on_loop, worker_terminal_id, loop)
        except RuntimeError:
            logger.debug("Assigned-worker retry loop closed during deadline cleanup")

    async def _retry_scheduler(self) -> None:
        """Wake only for a new failure or the nearest capped-backoff deadline."""
        loop = asyncio.get_running_loop()
        while True:
            with self._retry_lifecycle_guard:
                wakeup = self._retry_wakeup
            if wakeup is None:
                return

            if not self._retry_due:
                wakeup.clear()
                # No await occurs between the empty check and clear, so the
                # loop-owned scheduling callback cannot race through this gap.
                if not self._retry_due:
                    await wakeup.wait()
                continue

            next_due = min(self._retry_due.values())
            timeout = max(0.0, next_due - loop.time())
            wakeup.clear()
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=timeout)
                # A new/cleared deadline may be earlier; recompute from scratch.
                continue
            except TimeoutError:
                pass

            now = loop.time()
            ready = sorted(
                worker_id for worker_id, due_at in self._retry_due.items() if due_at <= now
            )
            for worker_terminal_id in ready:
                self._retry_due.pop(worker_terminal_id, None)
                try:
                    await asyncio.to_thread(self.reconcile_worker, worker_terminal_id)
                    record = await asyncio.to_thread(
                        get_assigned_worker_callback,
                        worker_terminal_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A database/backend exception before RETRYABLE could be
                    # persisted is still safe to re-run under the same capped
                    # backoff and per-worker serialization.
                    logger.exception(
                        "Assigned-worker delayed retry failed for %s",
                        worker_terminal_id,
                    )
                    self._schedule_retry_on_loop(worker_terminal_id, loop)
                    continue

                if record is not None and (
                    record.delivery_state == CompletionDeliveryState.RETRYABLE
                    or worker_terminal_id in self._failure_notice_pending
                ):
                    # Failure paths normally request this themselves. This check
                    # closes any exception/order seam without creating a duplicate
                    # deadline because scheduling is coalesced by worker id.
                    self._schedule_retry_on_loop(worker_terminal_id, loop)
                else:
                    self._retry_delays.pop(worker_terminal_id, None)

    def _mark_retryable(
        self,
        record: AssignedWorkerCallback,
        error: str,
        receiver_state: CompletionReceiverState = CompletionReceiverState.RETRYABLE_FAILURE,
    ) -> Optional[AssignedWorkerCallback]:
        """Persist retry state and wake the server-owned delayed scheduler."""
        updated = mark_completion_retryable(record.assignment_id, error, receiver_state)
        if updated is not None and updated.delivery_state == CompletionDeliveryState.RETRYABLE:
            self._request_retry(updated.worker_terminal_id)
        return updated

    async def run(self, registry: Optional[PluginRegistry] = None) -> None:
        """Consume status events; successful delivery requires no supervisor poll."""
        self._registry = registry
        queue = bus.subscribe("terminal.*.status")
        self.start_retry_scheduler()
        logger.info("AssignedWorkerCompletionService started")

        try:
            # Startup reconciliation remains mandatory, but now runs only after
            # the retry scheduler is able to retain any new transient failure.
            # It is inside the outer finally so cancellation during this first
            # database pass still tears down the scheduler.
            try:
                await asyncio.to_thread(self.reconcile_pending)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Assigned-worker startup reconciliation failed")

            while True:
                terminal_id: Optional[str] = None
                try:
                    event = await queue.get()
                    terminal_id = terminal_id_from_topic(event["topic"])
                    status = TerminalStatus(event["data"]["status"])
                    if status in (TerminalStatus.COMPLETED, TerminalStatus.ERROR):
                        await asyncio.to_thread(self.handle_status_event, terminal_id, status)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One corrupt event/receiver cannot stop completion delivery for
                    # every other assigned worker. A known terminal gets a bounded
                    # delayed retry even when the exception preceded durable state.
                    logger.exception("Assigned-worker completion event handling failed")
                    if terminal_id is not None:
                        self._request_retry(terminal_id)
        finally:
            await self.stop_retry_scheduler()

    def mark_dispatched_and_reconcile(self, worker_terminal_id: str) -> None:
        """Persist accepted task dispatch and close the fast-completion race.

        A very fast worker can reach COMPLETED before deferred initialization
        persists DISPATCHED.  The status consumer correctly ignores that early
        event; this method immediately re-checks the terminal after the durable
        dispatch write so the completion is not lost.
        """
        with self._worker_lock(worker_terminal_id):
            record = mark_assigned_worker_dispatched(worker_terminal_id)
            if record is None:
                return
            captured = self._capture_dispatched_result(record, retry_unavailable=False)
            if captured is not None and captured.lifecycle == AssignmentLifecycle.COMPLETED:
                self._release_capture_barrier(worker_terminal_id)
                self._drive_delivery(captured)
                return
            refreshed = get_assigned_worker_callback(worker_terminal_id)
            if refreshed is None or refreshed.delivery_state in _DELIVERY_TERMINAL_STATES:
                return
            record = refreshed
            status = self._detect_live_status(record)
            if status in (TerminalStatus.COMPLETED, TerminalStatus.ERROR):
                self._handle_status_locked(record, status)

    def handle_status_event(self, worker_terminal_id: str, status: TerminalStatus) -> None:
        """Handle a genuine provider-derived terminal status transition."""
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None:
                return
            self._handle_status_locked(record, status)

    def _handle_status_locked(self, record: AssignedWorkerCallback, status: TerminalStatus) -> None:
        if status == TerminalStatus.WAITING_USER_ANSWER:
            self._notify_worker_question(record)
            return
        if status == TerminalStatus.ERROR:
            # Commit a separate SYSTEM notice before transitioning the original
            # assignment. A follow-up failure never rewrites its successful report.
            if not self._notify_worker_failure(record):
                return
            if record.lifecycle != AssignmentLifecycle.COMPLETED:
                mark_completion_terminal_error(
                    record.assignment_id,
                    "Assigned worker reached terminal ERROR before successful completion",
                    CompletionReceiverState.UNKNOWN,
                    lifecycle=AssignmentLifecycle.FAILED,
                )
            self._release_capture_barrier(record.worker_terminal_id)
            return
        if status != TerminalStatus.COMPLETED:
            return

        # The dispatch gate prevents provider startup chrome or a stale previous
        # turn from being mistaken for completion of the newly assigned task.
        if record.lifecycle == AssignmentLifecycle.ASSIGNED:
            return
        if record.lifecycle in (AssignmentLifecycle.FAILED, AssignmentLifecycle.CANCELLED):
            return

        if record.lifecycle == AssignmentLifecycle.DISPATCHED:
            captured = self._capture_dispatched_result(record, retry_unavailable=True)
            if captured is None:
                return
            if captured.lifecycle != AssignmentLifecycle.COMPLETED:
                return
            record = captured

        # Once the final result is durable, later terminal input cannot destroy
        # it.  Callback enqueue may continue/retry independently.
        if record.lifecycle == AssignmentLifecycle.COMPLETED and record.final_result is not None:
            self._release_capture_barrier(record.worker_terminal_id)
        self._drive_delivery(record)

    def _notify_worker_question(self, record: AssignedWorkerCallback) -> None:
        """Wake the supervisor without answering or changing approval authority."""
        metadata = get_terminal_metadata(record.worker_terminal_id)
        if not metadata or metadata.get("provider") != "claude_code":
            return
        from cli_agent_orchestrator.services.claude_question import _screen, parse_screen

        try:
            screen = _screen(record.worker_terminal_id, record.caller_id)
            try:
                digest, _, _ = parse_screen(screen)
            except ValueError:
                # Owner/startup dialogs may not use a numbered question widget.
                digest = utf8_sha256(screen)
            create_inbox_message(
                record.worker_terminal_id,
                record.caller_id,
                "CAO worker " + record.worker_terminal_id + " is waiting for an answer. "
                "Inspect its current output. Answer only delegated routine questions with "
                "get_worker_question/answer_worker_question; forward owner-reserved or "
                "human-identity decisions to the owner. This notice is not approval.",
                origin=InboxMessageOrigin.SYSTEM,
                assignment_id=record.assignment_id,
                idempotency_key=f"assigned-worker-question:{record.completion_id}:{digest}",
            )
            self._attempt_immediate_inbox_delivery(record.caller_id)
        except (ValueError, RuntimeError):
            logger.warning("Could not inspect waiting worker %s", record.worker_terminal_id)

    def _capture_dispatched_result(
        self,
        record: AssignedWorkerCallback,
        *,
        retry_unavailable: bool,
    ) -> Optional[AssignedWorkerCallback]:
        """Persist one provider-authored report or fail closed.

        An absent report can be a normal race because Codex launches its notify
        command asynchronously and Claude's launcher persists before publishing
        the final stream event. A malformed, empty, conflicting, or wrongly
        correlated report is permanent for the immutable completion identity
        and therefore requires manual recovery rather than repeated guessing.
        """
        try:
            final_result = self._capture_final_result(record.worker_terminal_id)
            digest = utf8_sha256(final_result)
            return capture_assigned_worker_completion(
                record.worker_terminal_id,
                final_result,
                digest,
                f"assigned-worker-callback:{record.assignment_id}",
            )
        except ProviderCompletionUnavailableError as exc:
            if retry_unavailable:
                self._mark_retryable(
                    record,
                    f"Authoritative final report is not available yet: {exc}",
                    CompletionReceiverState.UNKNOWN,
                )
            return None
        except ProviderCompletionTerminalOutcomeError as exc:
            lifecycle = (
                AssignmentLifecycle.CANCELLED
                if exc.completion_state == "cancelled"
                else AssignmentLifecycle.FAILED
            )
            mark_completion_terminal_error(
                record.assignment_id,
                "Authoritative provider result was not successful: "
                f"state={exc.completion_state}, "
                f"subtype={exc.provider_result_subtype}, "
                f"terminal_reason={exc.provider_terminal_reason}",
                CompletionReceiverState.UNKNOWN,
                lifecycle=lifecycle,
            )
            self._release_capture_barrier(record.worker_terminal_id)
            return None
        except ProviderCompletionInvalidError as exc:
            logger.error(
                "Rejected authoritative completion for assigned worker %s: %s",
                record.worker_terminal_id,
                exc,
            )
            mark_assignment_manual_recovery(
                record.assignment_id,
                f"Authoritative final report rejected: {type(exc).__name__}: {exc}; "
                "terminal and report evidence retained for manual recovery",
            )
            return None
        except Exception as exc:
            logger.warning(
                "Could not capture authoritative final report for assigned worker %s: %s",
                record.worker_terminal_id,
                exc,
                exc_info=True,
            )
            self._mark_retryable(
                record,
                f"Authoritative final report capture failed: {type(exc).__name__}: {exc}",
                CompletionReceiverState.UNKNOWN,
            )
            return None

    @staticmethod
    def _capture_final_result(worker_terminal_id: str) -> str:
        """Return only a correlated provider-native final assistant response."""
        record = get_assigned_worker_callback(worker_terminal_id)
        if record is None:
            raise ProviderCompletionUnavailableError(
                f"assigned completion for terminal {worker_terminal_id} is unavailable"
            )
        provider = provider_manager.get_provider(worker_terminal_id)
        if provider is None:
            raise ProviderCompletionUnavailableError(
                f"provider for assigned terminal {worker_terminal_id} is unavailable"
            )
        report = provider.get_completion_report(record.completion_id)
        if not isinstance(report, ProviderCompletionReport):
            raise ProviderCompletionInvalidError(
                "provider adapter returned a non-contract completion report"
            )
        metadata = get_terminal_metadata(worker_terminal_id)
        expected_provider = metadata.get("provider") if metadata is not None else None
        if (
            report.provider != expected_provider
            or report.terminal_id != worker_terminal_id
            or report.completion_id != record.completion_id
        ):
            raise ProviderCompletionInvalidError(
                "provider report identity does not match the dispatched worker completion"
            )
        if report.completion_state != "success":
            raise ProviderCompletionTerminalOutcomeError(
                report.completion_state,
                report.provider_result_subtype,
                report.provider_terminal_reason,
            )
        if not report.final_response.strip():
            raise ProviderCompletionInvalidError(
                "provider report has no authoritative non-empty assistant response"
            )
        if report.final_response_sha256 != utf8_sha256(report.final_response):
            raise ProviderCompletionInvalidError("provider report response digest is invalid")
        return report.final_response

    def _classify_receiver(
        self, record: AssignedWorkerCallback
    ) -> Tuple[CompletionReceiverState, Optional[str]]:
        """Classify the immutable caller without changing or rerouting it."""
        if (
            not _TERMINAL_ID_RE.fullmatch(record.caller_id)
            or record.caller_id == record.worker_terminal_id
        ):
            return (
                CompletionReceiverState.PERMANENTLY_INVALID,
                f"Persisted caller_id is permanently invalid: {record.caller_id!r}",
            )

        try:
            metadata = get_terminal_metadata(record.caller_id)
        except Exception as exc:
            return (
                CompletionReceiverState.RETRYABLE_FAILURE,
                f"Receiver metadata lookup failed: {type(exc).__name__}: {exc}",
            )
        if metadata is None:
            return (
                CompletionReceiverState.DELETED,
                f"Persisted caller terminal {record.caller_id} was deleted",
            )

        try:
            backend = get_backend()
            if backend.supports_event_inbox():
                backend.get_pane_id(
                    record.caller_id,
                    metadata["tmux_session"],
                    metadata["tmux_window"],
                )
            else:
                if not backend.session_exists(metadata["tmux_session"]):
                    return CompletionReceiverState.RETAINED_UNREACHABLE, None
                # A successful history read positively resolves the retained
                # terminal/window.  Content may legitimately be empty.
                backend.get_history(metadata["tmux_session"], metadata["tmux_window"], tail_lines=1)
            return CompletionReceiverState.ACTIVE, None
        except (TerminalNotFoundError, ValueError):
            # The DB row remains a valid queue address even when its pane is
            # temporarily absent.  Enqueue once; inbox reconciliation can deliver
            # if the retained terminal becomes reachable again.
            return CompletionReceiverState.RETAINED_UNREACHABLE, None
        except TerminalBackendError as exc:
            return (
                CompletionReceiverState.RETRYABLE_FAILURE,
                f"Receiver backend classification failed: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return (
                CompletionReceiverState.RETRYABLE_FAILURE,
                f"Receiver classification failed: {type(exc).__name__}: {exc}",
            )

    def _drive_delivery(self, record: AssignedWorkerCallback) -> None:
        """Resume the durable enqueue/link/ack state machine idempotently."""
        if record.delivery_state in _DELIVERY_TERMINAL_STATES:
            return
        if record.lifecycle != AssignmentLifecycle.COMPLETED or record.final_result is None:
            return

        receiver_state, receiver_error = self._classify_receiver(record)
        if receiver_state in (
            CompletionReceiverState.DELETED,
            CompletionReceiverState.PERMANENTLY_INVALID,
        ):
            mark_completion_terminal_error(
                record.assignment_id,
                receiver_error or "Completion receiver is permanently unavailable",
                receiver_state,
            )
            return
        if receiver_state == CompletionReceiverState.RETRYABLE_FAILURE:
            self._mark_retryable(
                record,
                receiver_error or "Completion receiver classification is retryable",
                receiver_state,
            )
            return

        # V1 correction writes enqueue+link+ack atomically, but rows committed
        # by the earlier phased implementation may still be ENQUEUED after a
        # crash. Verify and acknowledge that exact immutable link; do not create
        # or paste a second row.
        if (
            record.delivery_state == CompletionDeliveryState.ENQUEUED
            and record.inbox_message_id is not None
        ):
            acknowledged = acknowledge_completion_enqueued(
                record.assignment_id, record.inbox_message_id
            )
            if acknowledged is not None:
                self._attempt_immediate_inbox_delivery(record.caller_id)
            return

        attempted = record_completion_delivery_attempt(record.assignment_id, receiver_state)
        if attempted is None or attempted.delivery_state in _DELIVERY_TERMINAL_STATES:
            return

        callback_message = self._format_callback_message(record)
        idempotency_key = f"assigned-worker-completion:{record.completion_id}"
        try:
            inbox_message = create_inbox_message(
                record.worker_terminal_id,
                record.caller_id,
                callback_message,
                origin=InboxMessageOrigin.SERVER_COMPLETION,
                assignment_id=record.assignment_id,
                idempotency_key=idempotency_key,
            )
            # Equivalence selection, inbox insert, callback link, and durable
            # acknowledgement now share one SQLite write transaction.  There is
            # no enqueue/link gap for an explicit sender to race into.
            self._phase_checkpoint("callback_committed", record, inbox_message.id)
            acknowledged = get_assigned_worker_callback(record.worker_terminal_id)
        except Exception as exc:
            logger.warning(
                "Completion enqueue for assignment %s is retryable: %s",
                record.assignment_id,
                exc,
                exc_info=True,
            )
            self._mark_retryable(
                record,
                f"Completion enqueue failed: {type(exc).__name__}: {exc}",
            )
            return

        if acknowledged is not None and acknowledged.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT,
        ):
            logger.info(
                "Committed assigned-worker completion %s as inbox row %s for caller %s (%s)",
                record.completion_id,
                inbox_message.id,
                record.caller_id,
                acknowledged.delivery_state.value,
            )
            self._attempt_immediate_inbox_delivery(record.caller_id)
            self._clear_requested_retry(record.worker_terminal_id)

    @staticmethod
    def _phase_checkpoint(
        phase: str, record: AssignedWorkerCallback, inbox_message_id: int
    ) -> None:
        """No-op fault-injection seam at committed recovery boundaries."""
        del phase, record, inbox_message_id

    @staticmethod
    def _format_callback_message(record: AssignedWorkerCallback) -> str:
        """Build the server-generated callback while preserving the report verbatim."""
        return format_server_completion_message(
            record.final_result or "",
            record.worker_terminal_id,
            record.assignment_id,
            record.completion_id,
        )

    def _attempt_immediate_inbox_delivery(self, caller_id: str) -> None:
        """Wake a ready caller; durable ack remains valid if this best-effort step fails."""
        try:
            from cli_agent_orchestrator.services.inbox_service import inbox_service

            inbox_service.deliver_pending(caller_id, registry=self._registry)
        except Exception:
            # The normal status-driven path and reconciliation daemon retain
            # responsibility for the already-durable PENDING row.
            logger.debug(
                "Immediate assigned-worker callback delivery failed for %s; inbox retry retained",
                caller_id,
                exc_info=True,
            )

    def _notify_worker_failure(self, record: AssignedWorkerCallback) -> bool:
        """Durable, idempotent failure notification; never a success callback."""
        if get_terminal_metadata(record.caller_id) is None:
            self._failure_notice_pending.discard(record.worker_terminal_id)
            return True
        message = (
            "CAO_WORKER_FAILED: worker="
            + record.worker_terminal_id
            + "; assignment="
            + record.assignment_id
            + ". The provider/transport entered ERROR. Do not wait for a reply or "
            "send more input to this terminal. Any previous final report is retained; "
            "the current turn is not proven complete. Inspect retained evidence and "
            "choose recovery explicitly. No retry or replacement has been started."
        )
        try:
            create_inbox_message(
                record.worker_terminal_id,
                record.caller_id,
                message,
                origin=InboxMessageOrigin.SYSTEM,
                assignment_id=record.assignment_id,
                idempotency_key=f"assigned-worker-failure:{record.completion_id}",
            )
        except Exception:
            self._failure_notice_pending.add(record.worker_terminal_id)
            self._request_retry(record.worker_terminal_id)
            logger.warning(
                "Worker failure notice enqueue deferred for %s", record.worker_terminal_id
            )
            return False
        self._failure_notice_pending.discard(record.worker_terminal_id)
        self._attempt_immediate_inbox_delivery(record.caller_id)
        return True

    def reconcile_worker(self, worker_terminal_id: str) -> None:
        """Reconcile one worker from durable state and live provider status."""
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None:
                return
            if worker_terminal_id in self._failure_notice_pending:
                self._handle_status_locked(record, TerminalStatus.ERROR)
                return
            if record.delivery_state in _DELIVERY_TERMINAL_STATES:
                return
            if record.lifecycle == AssignmentLifecycle.COMPLETED:
                self._drive_delivery(record)
                return
            if record.lifecycle in (AssignmentLifecycle.FAILED, AssignmentLifecycle.CANCELLED):
                return

            # The retained structured report is itself authoritative completion
            # evidence, so restart recovery does not depend on a pane still
            # rendering the old COMPLETED frame.  Missing reports fall through
            # to live status; malformed/cross-turn reports fail closed inside
            # _capture_dispatched_result.
            if record.lifecycle == AssignmentLifecycle.DISPATCHED:
                captured = self._capture_dispatched_result(record, retry_unavailable=True)
                if captured is not None and captured.lifecycle == AssignmentLifecycle.COMPLETED:
                    self._release_capture_barrier(worker_terminal_id)
                    self._drive_delivery(captured)
                    return
                refreshed = get_assigned_worker_callback(worker_terminal_id)
                if refreshed is None or refreshed.delivery_state in _DELIVERY_TERMINAL_STATES:
                    return
                record = refreshed

            status = self._detect_live_status(record)
            # Provider state after restart is not proof that the assigned prompt
            # was accepted: startup chrome or a stale previous turn can also be
            # PROCESSING/COMPLETED.  Only mark_dispatched_and_reconcile(), called
            # after the real input send returns, may open the completion gate.
            if record.lifecycle == AssignmentLifecycle.ASSIGNED:
                if status == TerminalStatus.ERROR:
                    self.record_worker_failure(
                        worker_terminal_id,
                        "Unproven assigned worker reached terminal ERROR after restart",
                    )
                elif status in (
                    TerminalStatus.PROCESSING,
                    TerminalStatus.COMPLETED,
                ):
                    mark_assignment_manual_recovery(
                        record.assignment_id,
                        "Restart observed live worker state without durable dispatch proof; "
                        "terminal retained for manual recovery",
                    )
                return
            if record.lifecycle == AssignmentLifecycle.UNRESOLVED:
                return
            if status in (TerminalStatus.COMPLETED, TerminalStatus.ERROR):
                self._handle_status_locked(record, status)

    def reconcile_pending(self) -> None:
        """Recover interrupted captures/enqueues without any supervisor polling."""
        # Recover failure delivery even when the original success callback was
        # already acknowledged. This scan runs only on service reconciliation,
        # never polls a model or restarts an assignment.
        for retained in list_retained_assigned_worker_callbacks():
            try:
                if self._detect_live_status(retained) == TerminalStatus.ERROR:
                    self.handle_status_event(retained.worker_terminal_id, TerminalStatus.ERROR)
            except Exception:
                logger.warning(
                    "Retained worker failure reconciliation deferred for %s",
                    retained.worker_terminal_id,
                )
        for record in list_reconcilable_assigned_worker_callbacks():
            try:
                self.register_assignment(record.worker_terminal_id)
                self.reconcile_worker(record.worker_terminal_id)
            except Exception:
                logger.exception(
                    "Assigned-worker callback reconciliation failed for %s",
                    record.worker_terminal_id,
                )
                self._request_retry(record.worker_terminal_id)

    def register_persisted_assignments(self) -> None:
        """Prime restart barriers before status/inbox consumers can observe readiness.

        This deliberately performs only the bounded callback-row read and
        in-memory registration.  Full provider/status reconciliation remains a
        background operation, but every unfinished assignment is protected before
        the server starts accepting or delivering terminal input.
        """
        try:
            records = list_protected_assigned_worker_callbacks()
        except OperationalError as exc:
            # Some embedded/test hosts intentionally replace init_db with a
            # partial bootstrap.  A missing callback table means there are no
            # persisted V1 assignments to protect; real init_db failures still
            # fail earlier during schema creation.
            if "no such table: assigned_worker_callbacks" not in str(exc).lower():
                raise
            logger.warning(
                "Assigned-worker callback table unavailable during startup priming",
                exc_info=True,
            )
            return
        for record in records:
            self.register_assignment(record.worker_terminal_id)

    @staticmethod
    def _detect_live_status(record: AssignedWorkerCallback) -> TerminalStatus:
        """Read provider status, falling back to live backend history after restart."""
        cached = status_monitor.get_status(record.worker_terminal_id)
        if cached != TerminalStatus.UNKNOWN:
            return cached
        metadata = get_terminal_metadata(record.worker_terminal_id)
        if metadata is None:
            return TerminalStatus.UNKNOWN
        try:
            provider = provider_manager.get_provider(record.worker_terminal_id)
            if provider is None:
                return TerminalStatus.UNKNOWN
            output = get_backend().get_history(
                metadata["tmux_session"], metadata["tmux_window"], full_history=True
            )
            return provider.get_status(output)
        except Exception:
            logger.debug(
                "Could not derive restart status for assigned worker %s",
                record.worker_terminal_id,
                exc_info=True,
            )
            return TerminalStatus.UNKNOWN

    def prepare_terminal_retirement(self, worker_terminal_id: str) -> bool:
        """Capture a completed report before teardown, or record true cancellation.

        Returns ``False`` whenever authoritative evidence is still retryable or
        requires manual recovery; callers must retain the worker. Once the
        report is captured, receiver delivery may continue after worker
        retirement because every needed response byte is in SQLite.
        """
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None:
                return True
            if record.lifecycle == AssignmentLifecycle.COMPLETED:
                self._drive_delivery(record)
                return True
            if record.lifecycle in (
                AssignmentLifecycle.FAILED,
                AssignmentLifecycle.CANCELLED,
            ):
                return True

            if record.lifecycle == AssignmentLifecycle.UNRESOLVED:
                return False

            status = self._detect_live_status(record)
            if record.lifecycle == AssignmentLifecycle.ASSIGNED:
                if status == TerminalStatus.ERROR:
                    self.record_worker_failure(
                        worker_terminal_id,
                        "Unproven assigned worker reached ERROR before retirement",
                    )
                    return True
                mark_assignment_manual_recovery(
                    record.assignment_id,
                    "Retirement requested without durable dispatch proof; terminal retained "
                    "for manual recovery",
                )
                return False

            # A retained provider report is stronger evidence than a pane that
            # has already returned to an idle prompt.  Recover it before
            # interpreting IDLE as cancellation during terminal teardown.  A
            # positive provider ERROR still wins: failed turns must never be
            # promoted to successful completions merely because stale or
            # contradictory report evidence exists.
            if (
                record.lifecycle == AssignmentLifecycle.DISPATCHED
                and status != TerminalStatus.ERROR
            ):
                captured = self._capture_dispatched_result(record, retry_unavailable=False)
                if captured is not None and captured.lifecycle == AssignmentLifecycle.COMPLETED:
                    self._release_capture_barrier(worker_terminal_id)
                    self._drive_delivery(captured)
                    return True

                refreshed = get_assigned_worker_callback(worker_terminal_id)
                if refreshed is None:
                    return True
                if refreshed.lifecycle == AssignmentLifecycle.UNRESOLVED:
                    return False
                # Preserve retryable capture failures as recoverable evidence;
                # teardown would otherwise erase the only handle available to
                # a later reconciliation pass.
                if refreshed.delivery_state == CompletionDeliveryState.RETRYABLE:
                    return False
                record = refreshed

            if status == TerminalStatus.COMPLETED:
                self._handle_status_locked(record, status)
                updated = get_assigned_worker_callback(worker_terminal_id)
                return bool(updated and updated.lifecycle == AssignmentLifecycle.COMPLETED)

            if status == TerminalStatus.UNKNOWN:
                self._mark_retryable(
                    record,
                    "Worker outcome is unknown; retirement deferred to preserve recovery handle",
                    CompletionReceiverState.UNKNOWN,
                )
                return False

            lifecycle = (
                AssignmentLifecycle.FAILED
                if status == TerminalStatus.ERROR
                else AssignmentLifecycle.CANCELLED
            )
            mark_completion_terminal_error(
                record.assignment_id,
                "Assigned worker retired before successful completion",
                CompletionReceiverState.UNKNOWN,
                lifecycle=lifecycle,
            )
            self._release_capture_barrier(worker_terminal_id)
            return True

    def record_worker_failure(self, worker_terminal_id: str, error: str) -> None:
        """Persist a non-success worker failure without generating a success callback."""
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None or record.lifecycle == AssignmentLifecycle.COMPLETED:
                return
            mark_completion_terminal_error(
                record.assignment_id,
                error,
                CompletionReceiverState.UNKNOWN,
                lifecycle=AssignmentLifecycle.FAILED,
            )
            self._release_capture_barrier(worker_terminal_id)

    def record_dispatch_uncertain(self, worker_terminal_id: str, error: str) -> None:
        """Fail closed when an input attempt began but acceptance is uncertain.

        Unlike a provider-initialize exception or exhausted status-confirmation
        retries, an exception during/after ``send_input`` is not positive proof
        that the task never started. An unproven assignment therefore becomes a
        retained manual-recovery record instead of being falsely failed/deleted.
        """
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None or record.lifecycle in (
                AssignmentLifecycle.COMPLETED,
                AssignmentLifecycle.FAILED,
                AssignmentLifecycle.CANCELLED,
                AssignmentLifecycle.UNRESOLVED,
            ):
                return
            if record.lifecycle == AssignmentLifecycle.ASSIGNED:
                mark_assignment_manual_recovery(record.assignment_id, error)
                return
            # DISPATCHED was already committed before a later reconciliation
            # exception. Preserve that proof and ask the active server scheduler
            # to re-evaluate the genuine provider state.
            self._request_retry(worker_terminal_id)


assigned_worker_completion_service = AssignedWorkerCompletionService()
