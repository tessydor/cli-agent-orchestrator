"""Delivers queued inbox messages when terminals become ready.

Consumer: terminal.{id}.status
"""

import asyncio
import logging
import threading
import uuid
from itertools import groupby

from cli_agent_orchestrator.backends.base import TerminalNotFoundError
from cli_agent_orchestrator.clients.database import (
    claim_inbox_message,
    get_pending_messages,
    is_assigned_worker_callback_inbox_message,
    list_pending_receiver_ids_by_provider,
    list_pending_receiver_ids_older_than,
    resolve_inbox_claim,
)
from cli_agent_orchestrator.constants import (
    EAGER_INBOX_DELIVERY,
    INBOX_RECONCILE_GRACE_SECONDS,
)
from cli_agent_orchestrator.models.inbox import MessageStatus, OrchestrationType
from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.plugins import PluginRegistry
from cli_agent_orchestrator.providers.manager import provider_manager
from cli_agent_orchestrator.services import terminal_service
from cli_agent_orchestrator.services.event_bus import bus
from cli_agent_orchestrator.services.status_monitor import status_monitor
from cli_agent_orchestrator.services.terminal_service import (
    NativeAcceptanceTimeoutError,
    TerminalCaptureNotDurableError,
)
from cli_agent_orchestrator.utils.event import terminal_id_from_topic

logger = logging.getLogger(__name__)


class InboxService:
    """Delivers one pending message per terminal per IDLE cycle."""

    def _delivery_lock(self, terminal_id: str) -> threading.RLock:
        """Return the shared per-terminal lock guarding native-input delivery.

        deliver_pending is read(PENDING) → status check → capture-barrier wait
        → mark DELIVERED → send, with no atomic claim at the DB layer, so two
        concurrent calls for the SAME terminal can both read the same oldest
        row before either marks it and deliver one task twice. Concurrent
        callers are real: the status-event consumer and the immediate POST
        path both dispatch to worker threads, and the OpenCode poller and the
        reconcile sweep add more.

        This used to be a private per-terminal Lock of InboxService's own,
        which only ever serialized InboxService's OWN deliveries against each
        other -- not against the other three send_input entry points
        (agent_step's initial assignment dispatch, the raw
        POST /terminals/{id}/input route, and agui's handoff-approval send).
        A follow-up delivered here could therefore still physically interleave
        its tmux paste with one of those (the proven correction-971 defect: a
        follow-up landing inside an assignment's still-open first-user paste,
        corrupting the dispatch-identity hash bind_completion_dispatch
        records). Using terminal_service's shared lock instead -- and holding
        it across wait_for_capture_before_input() too, not just around the
        eventual send_input() call -- closes the specific gap where that wait
        returns, the lock is momentarily not held, and a concurrent caller's
        send_input slips in before this delivery's own send_input call:
        extending the capture barrier's protection to every entry point
        uniformly, not just the physical tmux write. Safe because
        terminal_input_lock is an RLock and send_input (called from within
        this same held lock, on the same thread) reacquires it reentrantly.
        """
        return terminal_service.terminal_input_lock(terminal_id)

    async def run(self, registry: PluginRegistry | None = None) -> None:
        queue = bus.subscribe("terminal.*.status")
        logger.info("InboxService started")

        while True:
            try:
                event = await queue.get()
                status_value = event["data"]["status"]
                if status_value in (TerminalStatus.IDLE.value, TerminalStatus.COMPLETED.value):
                    terminal_id = terminal_id_from_topic(event["topic"])
                    # deliver_pending does blocking DB + tmux I/O. Offload it to a
                    # worker thread so this consumer keeps yielding to the event loop
                    # (StatusMonitor/LogWriter must not be starved — see the threading
                    # note in docs/event-driven-architecture.md). The registry is
                    # threaded through so status-driven deliveries fire
                    # PostSendMessageEvent hooks with the same attribution as the
                    # immediate and OpenCode-poller paths.
                    await asyncio.to_thread(self.deliver_pending, terminal_id, registry=registry)
            except Exception as e:
                logger.error(f"Error in InboxService: {e}")

    def deliver_pending(
        self,
        terminal_id: str,
        num_messages: int = 1,
        registry: PluginRegistry | None = None,
    ) -> None:
        """Deliver pending message(s) to a ready terminal. Use num_messages=0 for all.

        Status comes from the StatusMonitor (the event-driven source of truth).
        Delivery normally happens on IDLE/COMPLETED; providers that accept input
        mid-turn (``accepts_input_while_processing``) also receive messages while
        PROCESSING/WAITING_USER_ANSWER when ``EAGER_INBOX_DELIVERY`` is on (#251).
        When a plugin registry is supplied, the originating sender and a
        ``send_message`` orchestration type are threaded to ``terminal_service``
        so ``PostSendMessageEvent`` hooks fire with correct attribution.

        Safe to call from any thread: the whole read→mark→send sequence is
        serialized per terminal (see __init__ for why that is load-bearing).
        """
        # Plugin dispatch must never run while _delivery_lock (== terminal_
        # input_lock) is held (correction-1007 item 3 / message 1004):
        # send_input's own "dispatch after MY lock releases" (correction-994)
        # is a no-op here, because send_input's lock acquisition is a
        # REENTRANT re-acquire of the SAME RLock this method already holds --
        # the lock is still held by THIS method's own `with` block while
        # send_input's "after the lock" code runs. Collect deferred
        # dispatches (in original per-batch order) and run them only after
        # the lock below has genuinely, fully released.
        deferred_dispatches: list = []
        with self._delivery_lock(terminal_id):
            self._deliver_pending_locked(terminal_id, num_messages, registry, deferred_dispatches)
        for dispatch in deferred_dispatches:
            dispatch()

    def _deliver_pending_locked(
        self,
        terminal_id: str,
        num_messages: int,
        registry: PluginRegistry | None,
        deferred_dispatches: list,
    ) -> None:
        limit = num_messages if num_messages > 0 else 100
        messages = get_pending_messages(terminal_id, limit=limit)
        if not messages:
            return

        status = status_monitor.get_status(terminal_id)
        if status not in (TerminalStatus.IDLE, TerminalStatus.COMPLETED):
            # Not ready on the normal path. Eager delivery (#251) lets providers
            # that accept input mid-turn receive messages while PROCESSING or
            # WAITING_USER_ANSWER; only in that case do we need the provider.
            eager_eligible = False
            if EAGER_INBOX_DELIVERY and status in (
                TerminalStatus.PROCESSING,
                TerminalStatus.WAITING_USER_ANSWER,
            ):
                provider = provider_manager.get_provider(terminal_id)
                eager_eligible = provider is not None and getattr(
                    provider, "accepts_input_while_processing", False
                )
            if not eager_eligible:
                return

        # Consult the capture hold regardless of which eligible status got us
        # here (correction-1038). This used to be gated on
        # ``status == TerminalStatus.COMPLETED``, on the assumption that a
        # hold can only matter while a worker is COMPLETED -- true for a
        # live just-finished capture, but not for a hold reconstructed at
        # startup (correction-1033) for an unresolved persisted assignment:
        # its provider can legitimately report IDLE for an already-quiet
        # composer, which used to skip this check entirely. Unknown/non-
        # assigned/never-held terminals have no barrier and return
        # immediately (the overwhelming majority of calls); a timeout
        # deliberately leaves every row PENDING for the next reconciliation
        # pass instead of risking transcript loss or racing an unresolved
        # incident. ``send_input``'s own recheck below remains the true
        # authoritative boundary (994/1038) -- this is only the pre-claim
        # optimization that avoids claiming rows we already know aren't
        # deliverable yet.
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        if not assigned_worker_completion_service.wait_for_capture_before_input(terminal_id):
            logger.warning(
                "Deferred inbox delivery to worker %s until its unresolved capture "
                "hold is durably captured or explicitly acknowledged",
                terminal_id,
            )
            return

        # A stale read is harmless: every candidate must win a durable
        # PENDING -> DELIVERING compare-and-set before it may touch the pane.
        # Concurrent fast paths can both read the same row, but only one claim
        # token wins.  DELIVERING also closes re-entrant status-event delivery.
        claim_token = uuid.uuid4().hex
        claimed_messages = []
        try:
            for message in messages:
                claimed = claim_inbox_message(message.id, claim_token)
                if claimed is not None:
                    claimed_messages.append(claimed)
        except Exception:
            # No paste has occurred yet. Release earlier claims from this batch
            # so one corrupt/lost later candidate does not strand valid rows.
            for claimed in claimed_messages:
                resolve_inbox_claim(claimed.id, claim_token, MessageStatus.PENDING)
            raise
        messages = claimed_messages
        if not messages:
            return

        # Deliver in contiguous runs of the same sender. With the default
        # num_messages=1 this is a single run; when draining all pending messages
        # (num_messages=0) a batch can span multiple senders, so each run is sent
        # separately to keep PostSendMessageEvent attribution correct — otherwise
        # every message would be attributed to messages[0].sender_id.
        for sender_id, group in groupby(messages, key=lambda m: m.sender_id):
            batch = list(group)
            combined = "\n".join(m.message for m in batch)
            try:
                if registry is None:
                    terminal_service.send_input(terminal_id, combined)
                else:
                    terminal_service.send_input(
                        terminal_id,
                        combined,
                        registry=registry,
                        sender_id=sender_id,
                        orchestration_type=OrchestrationType.SEND_MESSAGE,
                        _defer_plugin_dispatch=deferred_dispatches,
                    )
            except TerminalNotFoundError as e:
                # Pane not resolvable yet (e.g. a herdr pane that isn't mapped
                # for this window). Treat as transient: reset to PENDING so the
                # reconcile sweep retries rather than marking FAILED. (#271 semantic.)
                for message in batch:
                    resolve_inbox_claim(message.id, claim_token, MessageStatus.PENDING)
                logger.warning(
                    f"Pane not resolvable for terminal {terminal_id}; leaving "
                    f"{len(batch)} message(s) pending for retry: {e}"
                )
            except NativeAcceptanceTimeoutError as e:
                # A PRIOR dispatch to this terminal (from any send_input entry
                # point) has not yet been confirmed accepted (correction-984).
                # Transient by construction: the next IDLE/COMPLETED status
                # event re-triggers deliver_pending, and by then the prior
                # turn will either have been accepted (fence cleared) or the
                # terminal genuinely never accepted it, which is not this
                # delivery's failure to own. Never treat as FAILED.
                for message in batch:
                    resolve_inbox_claim(message.id, claim_token, MessageStatus.PENDING)
                logger.warning(
                    f"Prior dispatch to terminal {terminal_id} not yet confirmed accepted; "
                    f"leaving {len(batch)} message(s) pending for retry: {e}"
                )
            except TerminalCaptureNotDurableError as e:
                # send_input's own authoritative recheck (994/1038) found an
                # unresolved capture hold -- either this terminal completed
                # its turn WHILE send_input was blocked inside the acceptance
                # wait (correction-994, this read's own earlier IDLE/COMPLETED
                # check could not have seen that), or a hold reconstructed at
                # startup (correction-1033) for an unresolved persisted
                # assignment is still armed despite a live IDLE read
                # (correction-1038). Transient either way: the next status
                # event re-triggers delivery, and by then the hold will
                # either be durable/released or the terminal has moved on.
                for message in batch:
                    resolve_inbox_claim(message.id, claim_token, MessageStatus.PENDING)
                logger.warning(
                    f"Terminal {terminal_id} has an unresolved capture hold; leaving "
                    f"{len(batch)} message(s) pending for retry: {e}"
                )
            except Exception as e:
                for message in batch:
                    # Completion callbacks have stronger durability semantics
                    # than ordinary inbox traffic.  A transient paste/backend
                    # failure must remain retryable after durable enqueue; this
                    # also covers an equivalent explicit final callback selected
                    # for duplicate suppression.  Unrelated legacy messages keep
                    # the established FAILED behavior.
                    is_assignment_callback = is_assigned_worker_callback_inbox_message(message.id)
                    retry_status = (
                        MessageStatus.PENDING if is_assignment_callback else MessageStatus.FAILED
                    )
                    logger.error(
                        "Failed to deliver message %s to %s; status reset to %s: %s",
                        message.id,
                        terminal_id,
                        retry_status.value,
                        e,
                    )
                    resolve_inbox_claim(message.id, claim_token, retry_status)
            else:
                logger.info(f"Delivered {len(batch)} message(s) to terminal {terminal_id}")
                for message in batch:
                    try:
                        resolved = resolve_inbox_claim(
                            message.id, claim_token, MessageStatus.DELIVERED
                        )
                    except Exception:
                        # Paste returned successfully. A resolution exception is
                        # beyond the observable side-effect boundary, so resetting
                        # would risk a duplicate paste. Retain DELIVERING/manual.
                        logger.exception(
                            "Could not resolve pasted inbox claim %s; manual recovery required",
                            message.id,
                        )
                        continue
                    if not resolved:
                        # The paste may already have occurred. Never blindly
                        # reset or redeliver a claim whose ownership/evidence
                        # changed; leave it for manual audit.
                        logger.error(
                            "Could not durably resolve delivered inbox claim %s; "
                            "manual recovery required",
                            message.id,
                        )

    def poll_opencode_pending_messages(self, registry: PluginRegistry | None = None) -> None:
        """Poll OpenCode terminals for pending inbox messages.

        OpenCode-specific wakeup path for providers whose pipe-pane logs do not
        change after the TUI settles, so the FIFO-driven StatusMonitor may not
        emit an IDLE/COMPLETED transition to trigger delivery on its own.
        """
        for terminal_id in list_pending_receiver_ids_by_provider(ProviderType.OPENCODE_CLI.value):
            try:
                self.deliver_pending(terminal_id, registry=registry)
            except Exception as e:
                logger.debug(f"OpenCode inbox poll failed for {terminal_id}: {e}")

    def reconcile_orphaned_messages(self, registry: PluginRegistry | None = None) -> None:
        """Re-attempt delivery for messages stuck in PENDING past the grace window.

        Provider-agnostic safety net for issue #131: when a receiving terminal is
        already idle, the immediate (on POST) delivery path may miss on a stale
        status, and an idle terminal produces no new output so the event-driven
        StatusMonitor never emits an IDLE/COMPLETED event to wake delivery —
        leaving the message orphaned. This sweep finds any such message and routes
        it back through the normal delivery gate (``deliver_pending``).

        Only messages older than ``INBOX_RECONCILE_GRACE_SECONDS`` are considered,
        so the sweep never competes with the fast paths for freshly queued
        messages — it only adopts ones they have already missed.
        """
        for terminal_id in list_pending_receiver_ids_older_than(INBOX_RECONCILE_GRACE_SECONDS):
            try:
                self.deliver_pending(terminal_id, registry=registry)
            except Exception as e:
                logger.debug(f"Inbox reconciliation failed for {terminal_id}: {e}")


inbox_service = InboxService()
