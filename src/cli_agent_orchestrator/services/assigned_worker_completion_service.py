"""Durable, model-independent assigned-worker completion callbacks.

The service consumes genuine terminal status transitions.  It never infers task
completion from arbitrary model text and never relies on the worker invoking
``send_message``.  Final output is captured before delivery, then driven through
an idempotent inbox enqueue/link/ack state machine that a startup reconciliation
can safely resume after any committed phase.
"""

import asyncio
import hashlib
import logging
import re
import threading
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


class AssignedWorkerCompletionService:
    """Capture and deliver successful assigned-worker terminal completions."""

    def __init__(self) -> None:
        self._registry: Optional[PluginRegistry] = None
        self._worker_locks: dict[str, threading.RLock] = {}
        self._worker_locks_guard = threading.Lock()
        self._known_workers: set[str] = set()
        self._capture_barriers: dict[str, threading.Event] = {}

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
            self._capture_barriers.setdefault(worker_terminal_id, threading.Event())

    def wait_for_capture_before_input(self, worker_terminal_id: str, timeout: float = 5.0) -> bool:
        """Return only when a known completed worker's report is safely durable.

        Unknown/non-assigned terminals have no barrier and retain the existing
        zero-wait inbox path.  On timeout the caller leaves the inbox row PENDING;
        reconciliation retries after capture rather than risking transcript loss.
        """
        with self._worker_locks_guard:
            barrier = self._capture_barriers.get(worker_terminal_id)
        return True if barrier is None else barrier.wait(timeout)

    def _release_capture_barrier(self, worker_terminal_id: str) -> None:
        with self._worker_locks_guard:
            barrier = self._capture_barriers.pop(worker_terminal_id, None)
            self._known_workers.discard(worker_terminal_id)
        if barrier is not None:
            barrier.set()

    async def run(self, registry: Optional[PluginRegistry] = None) -> None:
        """Consume status events; successful delivery requires no supervisor poll."""
        self._registry = registry
        queue = bus.subscribe("terminal.*.status")
        logger.info("AssignedWorkerCompletionService started")

        while True:
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
                # every other assigned worker.
                logger.exception("Assigned-worker completion event handling failed")

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
        if status == TerminalStatus.ERROR:
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
            try:
                final_result = self._capture_final_result(record.worker_terminal_id)
                digest = hashlib.sha256(final_result.encode("utf-8")).hexdigest()
                captured = capture_assigned_worker_completion(
                    record.worker_terminal_id,
                    final_result,
                    digest,
                    f"assigned-worker-callback:{record.assignment_id}",
                )
            except Exception as exc:  # output/backend failure is retryable
                logger.warning(
                    "Could not capture final report for assigned worker %s: %s",
                    record.worker_terminal_id,
                    exc,
                    exc_info=True,
                )
                mark_completion_retryable(
                    record.assignment_id,
                    f"Final report capture failed: {type(exc).__name__}: {exc}",
                    CompletionReceiverState.UNKNOWN,
                )
                return
            if captured is None or captured.lifecycle != AssignmentLifecycle.COMPLETED:
                return
            record = captured

        # Once the final result is durable, later terminal input cannot destroy
        # it.  Callback enqueue may continue/retry independently.
        if record.lifecycle == AssignmentLifecycle.COMPLETED and record.final_result is not None:
            self._release_capture_barrier(record.worker_terminal_id)
        self._drive_delivery(record)

    @staticmethod
    def _capture_final_result(worker_terminal_id: str) -> str:
        """Extract the provider's final response while its terminal is retained."""
        # Lazy import avoids terminal_service -> this service -> terminal_service
        # during the delete-time retirement hook.
        from cli_agent_orchestrator.services import terminal_service

        result = terminal_service.get_output(worker_terminal_id, terminal_service.OutputMode.LAST)
        if not isinstance(result, str):
            raise TypeError("provider final output was not text")
        return result

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
            mark_completion_retryable(
                record.assignment_id,
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
            mark_completion_retryable(
                record.assignment_id,
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

    def reconcile_worker(self, worker_terminal_id: str) -> None:
        """Reconcile one worker from durable state and live provider status."""
        with self._worker_lock(worker_terminal_id):
            record = get_assigned_worker_callback(worker_terminal_id)
            if record is None or record.delivery_state in _DELIVERY_TERMINAL_STATES:
                return
            if record.lifecycle == AssignmentLifecycle.COMPLETED:
                self._drive_delivery(record)
                return
            if record.lifecycle in (AssignmentLifecycle.FAILED, AssignmentLifecycle.CANCELLED):
                return

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
        for record in list_reconcilable_assigned_worker_callbacks():
            try:
                self.register_assignment(record.worker_terminal_id)
                self.reconcile_worker(record.worker_terminal_id)
            except Exception:
                logger.exception(
                    "Assigned-worker callback reconciliation failed for %s",
                    record.worker_terminal_id,
                )

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

        Returns ``False`` only when the terminal is visibly COMPLETED but final
        report capture could not be made durable; callers must retain the worker
        and allow a retry.  Once the report is captured, receiver delivery may
        continue after worker retirement because every needed byte is in SQLite.
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
            if status == TerminalStatus.COMPLETED:
                self._handle_status_locked(record, status)
                updated = get_assigned_worker_callback(worker_terminal_id)
                return bool(updated and updated.lifecycle == AssignmentLifecycle.COMPLETED)

            if status == TerminalStatus.UNKNOWN:
                mark_completion_retryable(
                    record.assignment_id,
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


assigned_worker_completion_service = AssignedWorkerCompletionService()
