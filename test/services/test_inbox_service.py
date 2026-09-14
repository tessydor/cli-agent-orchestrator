"""Tests for the event-driven InboxService."""

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Barrier, Lock, RLock
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.backends.base import TerminalNotFoundError
from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.constants import INBOX_RECONCILE_GRACE_SECONDS
from cli_agent_orchestrator.models.inbox import (
    InboxMessage,
    InboxMessageOrigin,
    MessageStatus,
    OrchestrationType,
)
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import inbox_service as inbox_mod
from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    AssignedWorkerCompletionService,
)
from cli_agent_orchestrator.services.inbox_service import InboxService


def _make_message(
    id=1,
    receiver_id="term-1",
    message="hello",
    status=MessageStatus.PENDING,
    sender_id="sender-1",
    **kwargs,
):
    return InboxMessage(
        id=id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message=message,
        status=status,
        created_at=datetime.now(),
        **kwargs,
    )


@pytest.fixture
def inbox_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'inbox-delivery.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        db,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    try:
        yield
    finally:
        engine.dispose()


def _delivery_fakes(monkeypatch, messages, status=TerminalStatus.IDLE):
    by_id = {message.id: message for message in messages}
    get_pending = MagicMock(return_value=messages)
    claim = MagicMock(
        side_effect=lambda message_id, token: by_id[message_id].model_copy(
            update={
                "status": MessageStatus.DELIVERING,
                "claim_token": token,
                "claimed_at": datetime.now(),
            }
        )
    )
    resolve = MagicMock(return_value=True)
    is_callback = MagicMock(return_value=False)
    terminal = MagicMock()
    monitor = MagicMock()
    monitor.get_status.return_value = status
    provider_manager = MagicMock()
    monkeypatch.setattr(inbox_mod, "get_pending_messages", get_pending)
    monkeypatch.setattr(inbox_mod, "claim_inbox_message", claim)
    monkeypatch.setattr(inbox_mod, "resolve_inbox_claim", resolve)
    monkeypatch.setattr(inbox_mod, "is_assigned_worker_callback_inbox_message", is_callback)
    monkeypatch.setattr(inbox_mod, "terminal_service", terminal)
    monkeypatch.setattr(inbox_mod, "status_monitor", monitor)
    monkeypatch.setattr(inbox_mod, "provider_manager", provider_manager)
    return SimpleNamespace(
        get=get_pending,
        claim=claim,
        resolve=resolve,
        is_callback=is_callback,
        terminal=terminal,
        monitor=monitor,
        provider_manager=provider_manager,
    )


class TestDeliverPending:
    @pytest.mark.parametrize("status", [TerminalStatus.IDLE, TerminalStatus.COMPLETED])
    def test_ready_terminal_claims_pastes_then_resolves(self, monkeypatch, status):
        fakes = _delivery_fakes(monkeypatch, [_make_message()], status)

        InboxService().deliver_pending("term-1")

        fakes.claim.assert_called_once_with(1, ANY)
        fakes.terminal.send_input.assert_called_once_with("term-1", "hello")
        fakes.resolve.assert_called_once_with(1, ANY, MessageStatus.DELIVERED)

    def test_completed_assigned_worker_waits_for_report_capture(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()], TerminalStatus.COMPLETED)
        with patch(
            "cli_agent_orchestrator.services.assigned_worker_completion_service."
            "assigned_worker_completion_service.wait_for_capture_before_input",
            return_value=False,
        ) as wait_for_capture:
            InboxService().deliver_pending("term-1")

        wait_for_capture.assert_called_once_with("term-1")
        fakes.claim.assert_not_called()
        fakes.terminal.send_input.assert_not_called()

    @pytest.mark.parametrize(
        "messages,status",
        [
            ([], TerminalStatus.IDLE),
            ([_make_message()], TerminalStatus.PROCESSING),
            ([_make_message()], TerminalStatus.UNKNOWN),
        ],
    )
    def test_unready_or_empty_queue_does_not_claim(self, monkeypatch, messages, status):
        fakes = _delivery_fakes(monkeypatch, messages, status)
        InboxService().deliver_pending("term-1")
        fakes.claim.assert_not_called()
        fakes.terminal.send_input.assert_not_called()

    def test_batches_contiguous_senders_and_preserves_attribution(self, monkeypatch):
        messages = [
            _make_message(id=1, sender_id="sender-a", message="one"),
            _make_message(id=2, sender_id="sender-a", message="two"),
            _make_message(id=3, sender_id="sender-b", message="three"),
        ]
        fakes = _delivery_fakes(monkeypatch, messages)
        registry = MagicMock()

        InboxService().deliver_pending("term-1", num_messages=0, registry=registry)

        fakes.get.assert_called_once_with("term-1", limit=100)
        # _defer_plugin_dispatch=[] (correction-1007 item 3): deliver_pending
        # always passes its own deferred-dispatch collector through to
        # send_input now, so plugin dispatch can be deferred past
        # deliver_pending's own outer lock rather than running while
        # send_input's REENTRANT inner lock acquisition merely re-enters it.
        assert fakes.terminal.send_input.call_args_list == [
            call(
                "term-1",
                "one\ntwo",
                registry=registry,
                sender_id="sender-a",
                orchestration_type=inbox_mod.OrchestrationType.SEND_MESSAGE,
                _defer_plugin_dispatch=[],
            ),
            call(
                "term-1",
                "three",
                registry=registry,
                sender_id="sender-b",
                orchestration_type=inbox_mod.OrchestrationType.SEND_MESSAGE,
                _defer_plugin_dispatch=[],
            ),
        ]
        assert fakes.resolve.call_count == 3

    def test_atomic_claim_precedes_paste_and_resolution(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        order = []
        real_claim = fakes.claim.side_effect
        fakes.claim.side_effect = lambda *args: (order.append("claim"), real_claim(*args))[1]
        fakes.terminal.send_input.side_effect = lambda *args: order.append("paste")
        fakes.resolve.side_effect = lambda *args: order.append("resolve") or True

        InboxService().deliver_pending("term-1")

        assert order == ["claim", "paste", "resolve"]

    def test_losing_claim_never_pastes(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        fakes.claim.return_value = None
        fakes.claim.side_effect = None
        InboxService().deliver_pending("term-1")
        fakes.terminal.send_input.assert_not_called()
        fakes.resolve.assert_not_called()

    def test_later_claim_failure_releases_earlier_claim_before_any_paste(self, monkeypatch):
        fakes = _delivery_fakes(
            monkeypatch,
            [_make_message(id=1), _make_message(id=2, message="second")],
        )
        normal_claim = fakes.claim.side_effect

        def fail_second_claim(message_id, token):
            if message_id == 2:
                raise RuntimeError("corrupt second row")
            return normal_claim(message_id, token)

        fakes.claim.side_effect = fail_second_claim

        with pytest.raises(RuntimeError, match="corrupt second row"):
            InboxService().deliver_pending("term-1", num_messages=0)

        fakes.terminal.send_input.assert_not_called()
        fakes.resolve.assert_called_once_with(1, ANY, MessageStatus.PENDING)

    @pytest.mark.parametrize(
        "is_callback,expected", [(False, MessageStatus.FAILED), (True, MessageStatus.PENDING)]
    )
    def test_caught_send_failure_resolves_claim_for_retry_policy(
        self, monkeypatch, is_callback, expected
    ):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        fakes.is_callback.return_value = is_callback
        fakes.terminal.send_input.side_effect = RuntimeError("backend unavailable")

        InboxService().deliver_pending("term-1")

        fakes.resolve.assert_called_once_with(1, ANY, expected)

    def test_terminal_resolution_failure_resets_claim_to_pending(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        fakes.terminal.send_input.side_effect = TerminalNotFoundError("s:w")
        InboxService().deliver_pending("term-1")
        fakes.resolve.assert_called_once_with(1, ANY, MessageStatus.PENDING)

    def test_post_paste_claim_resolution_failure_is_manual_not_blind_reset(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        fakes.resolve.return_value = False
        InboxService().deliver_pending("term-1")
        fakes.terminal.send_input.assert_called_once()
        fakes.resolve.assert_called_once_with(1, ANY, MessageStatus.DELIVERED)

    def test_post_paste_claim_resolution_exception_is_manual_not_blind_reset(self, monkeypatch):
        fakes = _delivery_fakes(monkeypatch, [_make_message()])
        fakes.resolve.side_effect = RuntimeError("database unavailable after paste")

        InboxService().deliver_pending("term-1")

        fakes.terminal.send_input.assert_called_once()
        fakes.resolve.assert_called_once_with(1, ANY, MessageStatus.DELIVERED)
        fakes.is_callback.assert_not_called()

    def test_concurrent_delivery_for_one_terminal_sends_exactly_once(self, inbox_db, monkeypatch):
        caller, message = _create_delivery_row(False)
        monkeypatch.setattr(inbox_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE)
        send = MagicMock()
        monkeypatch.setattr(inbox_mod.terminal_service, "send_input", send)
        service = InboxService()
        start = Barrier(2)

        def race():
            start.wait(timeout=5)
            service.deliver_pending(caller)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(race) for _ in range(2)]
            for future in futures:
                future.result(timeout=10)
        send.assert_called_once()
        assert db.get_inbox_messages(caller, limit=10)[0].status == MessageStatus.DELIVERED


class TestEagerInboxDelivery:
    @pytest.mark.parametrize(
        "status,eager,capable,expected",
        [
            (TerminalStatus.IDLE, False, False, True),
            (TerminalStatus.COMPLETED, False, False, True),
            (TerminalStatus.PROCESSING, True, True, True),
            (TerminalStatus.PROCESSING, True, False, False),
            (TerminalStatus.PROCESSING, False, True, False),
            (TerminalStatus.WAITING_USER_ANSWER, True, True, True),
            (TerminalStatus.ERROR, True, True, False),
        ],
    )
    def test_eager_delivery_gate(self, monkeypatch, status, eager, capable, expected):
        fakes = _delivery_fakes(monkeypatch, [_make_message()], status)
        provider = MagicMock()
        provider.accepts_input_while_processing = capable
        fakes.provider_manager.get_provider.return_value = provider
        monkeypatch.setattr(inbox_mod, "EAGER_INBOX_DELIVERY", eager)

        InboxService().deliver_pending("term-1")

        assert fakes.terminal.send_input.called is expected


def _create_delivery_row(server_callback: bool):
    caller = "11111111"
    worker = "22222222"
    db.create_terminal(caller, "cao-inbox", "caller", "mock_cli")
    if not server_callback:
        return caller, db.create_inbox_message(worker, caller, "ordinary message")
    db.create_terminal(
        worker,
        "cao-inbox",
        "worker",
        "mock_cli",
        caller_id=caller,
        assignment_id="assignment-concurrent-delivery",
        completion_id="completion-concurrent-delivery",
    )
    db.mark_assigned_worker_dispatched(worker)
    report = "server final"
    captured = db.capture_assigned_worker_completion(
        worker,
        report,
        hashlib.sha256(report.encode()).hexdigest(),
        "assigned-worker-callback:assignment-concurrent-delivery",
    )
    assert captured is not None
    return caller, db.create_inbox_message(
        worker,
        caller,
        AssignedWorkerCompletionService._format_callback_message(captured),
        origin=InboxMessageOrigin.SERVER_COMPLETION,
        assignment_id=captured.assignment_id,
        idempotency_key=f"assigned-worker-completion:{captured.completion_id}",
    )


@pytest.mark.parametrize("server_callback", [False, True])
def test_concurrent_delivery_has_one_durable_paste(inbox_db, monkeypatch, server_callback):
    """The DB-level claim (claim_inbox_message) is the ultimate guarantor of
    exactly-once delivery -- defense in depth, independent of any in-process
    lock. _delivery_lock is forced to a FRESH, unshared RLock per call (via
    two separate InboxService instances plus this patch) specifically to
    remove in-process serialization from the picture: since correction-971,
    _delivery_lock returns terminal_service's module-level shared per-
    terminal lock, which now correctly serializes deliver_pending calls
    across instances too (see test_native_input_serialization_race.py for
    that guarantee) -- so without this patch the two threads below would
    never race at the read in the first place. This test intentionally
    forces the race anyway to prove the DB claim alone -- not merely the
    lock -- prevents a duplicate paste.
    """
    caller, message = _create_delivery_row(server_callback)
    real_get = db.get_pending_messages
    both_read_pending = Barrier(2)

    def racing_read(receiver_id, limit=1):
        rows = real_get(receiver_id, limit=limit)
        both_read_pending.wait(timeout=3)
        return rows

    monkeypatch.setattr(inbox_mod, "get_pending_messages", racing_read)
    monkeypatch.setattr(inbox_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE)
    monkeypatch.setattr(InboxService, "_delivery_lock", lambda self, terminal_id: RLock())
    pastes = []
    paste_lock = Lock()

    def record_paste(*args, **kwargs):
        with paste_lock:
            pastes.append((args, kwargs))

    monkeypatch.setattr(inbox_mod.terminal_service, "send_input", record_paste)
    service = InboxService()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(InboxService().deliver_pending, caller) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)

    assert len(pastes) == 1
    stored = db.get_inbox_messages(caller, limit=10)
    assert [row.id for row in stored] == [message.id]
    assert stored[0].status == MessageStatus.DELIVERED


def test_deferred_follow_up_never_overlaps_a_concurrent_assign_dispatch(monkeypatch):
    """Correction-971/984's real-world shape, end to end across both modules.

    An ASSIGN-orchestration send_input() call made directly (standing in for
    agent_step.py's initial-dispatch entry point, the one InboxService's own
    private lock never shared anything with pre-971-fix) dispatches first;
    its native acceptance is then deliberately, explicitly confirmed (the
    real StatusMonitor detection loop's role, simulated here); only THEN does
    InboxService.deliver_pending attempt the completed worker's queued
    follow-up. Deliberate ordering, not a symmetric race -- correction-984
    found that a symmetric release proves only non-overlap, not the correct
    order a real incident (assignment first, follow-up after) requires. Both
    calls run the REAL InboxService.deliver_pending / terminal_service.
    send_input code (only DB/tmux/provider edges are mocked), so this proves
    the shared lock AND the native-acceptance fence both actually apply
    across the two modules -- not merely within one.
    """
    terminal_id = "shared-completed-worker"
    message = _make_message(id=42, receiver_id=terminal_id, message="queued follow-up")
    monkeypatch.setattr(inbox_mod, "get_pending_messages", MagicMock(return_value=[message]))
    monkeypatch.setattr(
        inbox_mod,
        "claim_inbox_message",
        MagicMock(
            return_value=message.model_copy(
                update={
                    "status": MessageStatus.DELIVERING,
                    "claim_token": "tok",
                    "claimed_at": datetime.now(),
                }
            )
        ),
    )
    monkeypatch.setattr(inbox_mod, "resolve_inbox_claim", MagicMock(return_value=True))
    monkeypatch.setattr(
        inbox_mod, "is_assigned_worker_callback_inbox_message", MagicMock(return_value=False)
    )
    # status_monitor is the SAME singleton imported into both inbox_service and
    # terminal_service -- patching it here makes both modules' checks agree,
    # matching the real deployment.
    monkeypatch.setattr(
        inbox_mod.status_monitor, "get_status", MagicMock(return_value=TerminalStatus.COMPLETED)
    )
    from cli_agent_orchestrator.services.assigned_worker_completion_service import (
        assigned_worker_completion_service,
    )

    # The capture barrier has already cleared by the time both callers race --
    # this isolates the remaining gap correction-971 actually found (nothing
    # serializing the recheck-then-send step against a concurrent caller),
    # rather than re-testing the barrier itself (already covered elsewhere).
    monkeypatch.setattr(
        assigned_worker_completion_service, "wait_for_capture_before_input", lambda _id: True
    )

    from cli_agent_orchestrator.services import terminal_service as real_terminal_service

    monkeypatch.setattr(
        real_terminal_service,
        "get_terminal_metadata",
        MagicMock(
            return_value={
                "tmux_session": "cao-session",
                "tmux_window": "developer-shared",
                "provider": "codex",
            }
        ),
    )
    monkeypatch.setattr(real_terminal_service, "update_last_active", MagicMock())
    monkeypatch.setattr(
        real_terminal_service,
        "inject_memory_context",
        lambda message, terminal_id, frozen_memory=None: message,
    )
    callback = MagicMock()
    callback.completion_id = "1" * 32
    monkeypatch.setattr(
        real_terminal_service, "get_assigned_worker_callback", MagicMock(return_value=callback)
    )
    mock_bind = MagicMock()
    monkeypatch.setattr(
        "cli_agent_orchestrator.services.provider_completion_report.bind_completion_dispatch",
        mock_bind,
    )
    provider = MagicMock()
    provider.paste_enter_count = 2
    provider.paste_submit_delay = 0.0
    # provider_manager is the real shared singleton (also imported
    # independently into status_monitor.py) -- a raw attribute assignment
    # here would leak a stub get_provider into every later test in the
    # session (this was found causing cross-file pollution of
    # test_status_monitor.py). Use monkeypatch so it is restored.
    monkeypatch.setattr(
        real_terminal_service.provider_manager, "get_provider", MagicMock(return_value=provider)
    )

    occupancy = 0
    max_seen = 0
    occ_guard = Lock()
    writes = []

    def send_keys_side_effect(session, window, message, **kwargs):
        nonlocal occupancy, max_seen
        with occ_guard:
            occupancy += 1
            max_seen = max(max_seen, occupancy)
            writes.append(message)
        import time as _time

        _time.sleep(0.02)
        with occ_guard:
            occupancy -= 1
        return None

    try:
        with patch("cli_agent_orchestrator.backends.registry._backend") as mock_backend:
            mock_backend.send_keys.side_effect = send_keys_side_effect
            assign_text = "assigned task bytes"

            # Deliberate order (correction-984): the assignment dispatches
            # first and completes...
            real_terminal_service.send_input(
                terminal_id, assign_text, orchestration_type=OrchestrationType.ASSIGN
            )
            assert writes == [assign_text]

            # ...its native acceptance is then explicitly, deliberately
            # confirmed -- simulating StatusMonitor's real detection loop --
            # releasing the fence armed by that dispatch's own
            # notify_input_sent (called internally by send_input above).
            # correction-1001: a missing freshness timestamp now fails
            # CLOSED, so this simulated "real detection" must supply one,
            # exactly as a genuine _process_chunk-driven detection would.
            inbox_mod.status_monitor._buffer_changed_at[terminal_id] = time.monotonic()
            inbox_mod.status_monitor._apply_detection(terminal_id, TerminalStatus.PROCESSING)

            # ...only THEN does the completed worker's queued follow-up
            # attempt delivery.
            InboxService().deliver_pending(terminal_id)
    finally:
        inbox_mod.status_monitor.clear_terminal(terminal_id)

    assert max_seen == 1, (
        f"follow-up and ASSIGN dispatch tmux writes overlapped ({max_seen} concurrent) "
        "-- correction-971's cross-module race reproduced"
    )
    assert writes == [
        assign_text,
        "queued follow-up",
    ], f"expected exact order [assign, follow-up], got {writes!r}"
    mock_bind.assert_called_once_with("codex", terminal_id, "1" * 32, assign_text)


def test_deliver_pending_plugin_dispatch_finds_the_outer_lock_already_released(monkeypatch):
    """Correction-1007 item 3 / message 1004: moving PostSendMessageEvent
    dispatch outside send_input()'s OWN ``with terminal_input_lock(...):``
    block (correction-994) is insufficient when InboxService.deliver_pending
    already holds that SAME per-terminal RLock around
    ``_deliver_pending_locked -> send_input``. send_input's own lock
    acquisition there is a REENTRANT re-acquire of the lock deliver_pending
    already holds, so the lock is still held by deliver_pending's own
    ``with`` block while send_input's "after the lock" code runs.

    This exercises the REAL ``InboxService.deliver_pending ->
    _deliver_pending_locked -> terminal_service.send_input`` call chain
    (only DB/tmux/provider edges are mocked) with a genuinely cross-thread
    lock probe -- direct send_input-only coverage (see
    test_native_input_serialization_race.py's own
    ``test_plugin_dispatch_finds_the_lock_already_released``) cannot see
    this defect, because that test never nests send_input inside an outer
    holder of the same lock the way deliver_pending does.
    """
    terminal_id = "inbox-plugin-dispatch-terminal"
    message = _make_message(id=99, receiver_id=terminal_id, message="queued follow-up")
    monkeypatch.setattr(inbox_mod, "get_pending_messages", MagicMock(return_value=[message]))
    monkeypatch.setattr(
        inbox_mod,
        "claim_inbox_message",
        MagicMock(
            return_value=message.model_copy(
                update={
                    "status": MessageStatus.DELIVERING,
                    "claim_token": "tok",
                    "claimed_at": datetime.now(),
                }
            )
        ),
    )
    monkeypatch.setattr(inbox_mod, "resolve_inbox_claim", MagicMock(return_value=True))
    monkeypatch.setattr(
        inbox_mod, "is_assigned_worker_callback_inbox_message", MagicMock(return_value=False)
    )
    monkeypatch.setattr(
        inbox_mod.status_monitor, "get_status", MagicMock(return_value=TerminalStatus.IDLE)
    )

    from cli_agent_orchestrator.services import terminal_service as real_terminal_service

    monkeypatch.setattr(
        real_terminal_service,
        "get_terminal_metadata",
        MagicMock(
            return_value={
                "tmux_session": "cao-session",
                "tmux_window": "developer-plugin",
                "provider": "codex",
            }
        ),
    )
    monkeypatch.setattr(real_terminal_service, "update_last_active", MagicMock())
    monkeypatch.setattr(
        real_terminal_service,
        "inject_memory_context",
        lambda message, terminal_id, frozen_memory=None: message,
    )
    provider = MagicMock()
    provider.paste_enter_count = 1
    provider.paste_submit_delay = 0.0
    # provider_manager is the real shared singleton -- monkeypatch (not a
    # raw assignment) so it is restored; see the sibling test above.
    monkeypatch.setattr(
        real_terminal_service.provider_manager, "get_provider", MagicMock(return_value=provider)
    )

    events: list[tuple] = []
    writes: list[str] = []

    def send_keys_side_effect(*args, **kwargs):
        writes.append("sent")
        return None

    class _FakeRegistry:
        async def dispatch(self, event_type, event):
            # dispatch_plugin_event's no-loop branch runs THIS coroutine
            # synchronously (asyncio.run) on deliver_pending's OWN thread --
            # exactly the case that, pre-fix, ran while InboxService.
            # deliver_pending's OWN _delivery_lock (== terminal_input_lock)
            # was still held. Probe from a genuinely separate thread; a
            # same-thread check would be meaningless under RLock
            # reentrance (it would report "free" either way).
            lock = real_terminal_service.terminal_input_lock(terminal_id)
            with ThreadPoolExecutor(max_workers=1) as probe_pool:
                acquired = probe_pool.submit(lock.acquire, True, 1.0).result(timeout=5)
            events.append(("dispatch", event_type, acquired))
            if acquired:
                lock.release()

    try:
        with patch("cli_agent_orchestrator.backends.registry._backend") as mock_backend:
            mock_backend.send_keys.side_effect = send_keys_side_effect
            InboxService().deliver_pending(terminal_id, registry=_FakeRegistry())
    finally:
        inbox_mod.status_monitor.clear_terminal(terminal_id)

    assert writes == ["sent"]
    assert events == [
        ("dispatch", "post_send_message", True)
    ], f"expected exactly one dispatch, with the OUTER deliver_pending lock free; got {events!r}"


@pytest.mark.parametrize(
    "server_callback,expected_after_failure,expected_pastes",
    [(True, MessageStatus.PENDING, 2), (False, MessageStatus.FAILED, 1)],
)
def test_failure_reset_and_retry_policy_with_real_claims(
    inbox_db,
    monkeypatch,
    server_callback,
    expected_after_failure,
    expected_pastes,
):
    caller, _message = _create_delivery_row(server_callback)
    monkeypatch.setattr(inbox_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE)
    paste_count = 0

    def flaky_paste(*_args, **_kwargs):
        nonlocal paste_count
        paste_count += 1
        if paste_count == 1:
            raise RuntimeError("synthetic caught send failure")

    monkeypatch.setattr(inbox_mod.terminal_service, "send_input", flaky_paste)
    service = InboxService()
    service.deliver_pending(caller)
    assert db.get_inbox_messages(caller, limit=10)[0].status == expected_after_failure
    service.deliver_pending(caller)
    assert paste_count == expected_pastes
    expected_final = MessageStatus.DELIVERED if server_callback else MessageStatus.FAILED
    assert db.get_inbox_messages(caller, limit=10)[0].status == expected_final


@pytest.mark.parametrize("server_callback", [False, True])
def test_unresolved_delivering_claim_is_not_automatically_replayed(
    inbox_db,
    monkeypatch,
    server_callback,
):
    caller, message = _create_delivery_row(server_callback)
    assert db.claim_inbox_message(message.id, "crash-boundary-claim") is not None
    monkeypatch.setattr(inbox_mod.status_monitor, "get_status", lambda _id: TerminalStatus.IDLE)
    send_input = MagicMock()
    monkeypatch.setattr(inbox_mod.terminal_service, "send_input", send_input)

    InboxService().deliver_pending(caller)

    send_input.assert_not_called()
    stored = db.get_inbox_messages(caller, limit=10)[0]
    assert stored.status == MessageStatus.DELIVERING
    assert stored.claim_token == "crash-boundary-claim"


class TestPollOpenCodePendingMessages:
    """Tests for the OpenCode inbox poller."""

    @patch("cli_agent_orchestrator.services.inbox_service.list_pending_receiver_ids_by_provider")
    def test_polls_pending_opencode_receivers(self, mock_list_receivers):
        """Test poller attempts delivery for each pending OpenCode receiver."""
        mock_list_receivers.return_value = ["receiver-1", "receiver-2"]

        svc = InboxService()
        svc.deliver_pending = MagicMock()
        svc.poll_opencode_pending_messages()

        mock_list_receivers.assert_called_once_with("opencode_cli")
        assert svc.deliver_pending.call_args_list == [
            call("receiver-1", registry=None),
            call("receiver-2", registry=None),
        ]

    @patch("cli_agent_orchestrator.services.inbox_service.list_pending_receiver_ids_by_provider")
    def test_survives_per_receiver_failure(self, mock_list_receivers):
        """Test one failed receiver does not stop the poll loop."""
        mock_list_receivers.return_value = ["receiver-1", "receiver-2"]

        svc = InboxService()
        svc.deliver_pending = MagicMock(side_effect=[Exception("tmux busy"), None])
        svc.poll_opencode_pending_messages()

        assert svc.deliver_pending.call_count == 2


class TestReconcileOrphanedMessages:
    """Tests for the provider-agnostic inbox reconciliation sweep (issue #131)."""

    @patch("cli_agent_orchestrator.services.inbox_service.list_pending_receiver_ids_older_than")
    def test_reconciles_stale_receivers(self, mock_list_receivers):
        """Sweep attempts delivery for each receiver with an orphaned message."""
        mock_list_receivers.return_value = ["receiver-1", "receiver-2"]

        svc = InboxService()
        svc.deliver_pending = MagicMock()
        svc.reconcile_orphaned_messages()

        mock_list_receivers.assert_called_once_with(INBOX_RECONCILE_GRACE_SECONDS)
        assert svc.deliver_pending.call_args_list == [
            call("receiver-1", registry=None),
            call("receiver-2", registry=None),
        ]

    @patch("cli_agent_orchestrator.services.inbox_service.list_pending_receiver_ids_older_than")
    def test_survives_per_receiver_failure(self, mock_list_receivers):
        """One failed receiver does not stop the sweep."""
        mock_list_receivers.return_value = ["receiver-1", "receiver-2"]

        svc = InboxService()
        svc.deliver_pending = MagicMock(side_effect=[Exception("tmux busy"), None])
        svc.reconcile_orphaned_messages()

        assert svc.deliver_pending.call_count == 2


class TestRun:
    """Tests for InboxService.run() event loop."""

    @pytest.mark.asyncio
    async def test_processes_idle_status_event(self):
        svc = InboxService()
        svc.deliver_pending = MagicMock()

        queue = asyncio.Queue()
        await queue.put(
            {
                "topic": "terminal.abc123.status",
                "data": {"status": TerminalStatus.IDLE.value},
            }
        )

        with patch("cli_agent_orchestrator.services.inbox_service.bus") as mock_bus:
            mock_bus.subscribe.return_value = queue

            # Run one iteration then cancel
            async def run_one():
                task = asyncio.create_task(svc.run())
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_one()

        svc.deliver_pending.assert_called_once_with("abc123", registry=None)

    @pytest.mark.asyncio
    async def test_processes_completed_status_event(self):
        svc = InboxService()
        svc.deliver_pending = MagicMock()

        queue = asyncio.Queue()
        await queue.put(
            {
                "topic": "terminal.xyz789.status",
                "data": {"status": TerminalStatus.COMPLETED.value},
            }
        )

        with patch("cli_agent_orchestrator.services.inbox_service.bus") as mock_bus:
            mock_bus.subscribe.return_value = queue

            task = asyncio.create_task(svc.run())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        svc.deliver_pending.assert_called_once_with("xyz789", registry=None)

    @pytest.mark.asyncio
    async def test_ignores_processing_status_event(self):
        svc = InboxService()
        svc.deliver_pending = MagicMock()

        queue = asyncio.Queue()
        await queue.put(
            {
                "topic": "terminal.abc123.status",
                "data": {"status": TerminalStatus.PROCESSING.value},
            }
        )

        with patch("cli_agent_orchestrator.services.inbox_service.bus") as mock_bus:
            mock_bus.subscribe.return_value = queue

            task = asyncio.create_task(svc.run())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        svc.deliver_pending.assert_not_called()

    @pytest.mark.asyncio
    async def test_threads_registry_to_delivery(self):
        """run(registry) threads the plugin registry to deliver_pending so
        status-driven deliveries fire PostSendMessageEvent hooks with the same
        attribution as the immediate and OpenCode-poller paths (PR #273 review).
        """
        svc = InboxService()
        svc.deliver_pending = MagicMock()
        registry = MagicMock()

        queue = asyncio.Queue()
        await queue.put(
            {
                "topic": "terminal.abc123.status",
                "data": {"status": TerminalStatus.IDLE.value},
            }
        )

        with patch("cli_agent_orchestrator.services.inbox_service.bus") as mock_bus:
            mock_bus.subscribe.return_value = queue

            task = asyncio.create_task(svc.run(registry))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        svc.deliver_pending.assert_called_once_with("abc123", registry=registry)

    @pytest.mark.asyncio
    async def test_offloads_delivery_to_thread(self):
        """Delivery is offloaded via asyncio.to_thread so the consumer loop keeps
        yielding to the event loop and never blocks StatusMonitor/LogWriter on
        deliver_pending's synchronous DB + tmux I/O (PR #273 review; see the
        threading discipline note in docs/event-driven-architecture.md).
        """
        svc = InboxService()
        svc.deliver_pending = MagicMock()

        queue = asyncio.Queue()
        await queue.put(
            {
                "topic": "terminal.abc123.status",
                "data": {"status": TerminalStatus.IDLE.value},
            }
        )

        with (
            patch("cli_agent_orchestrator.services.inbox_service.bus") as mock_bus,
            patch(
                "cli_agent_orchestrator.services.inbox_service.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread,
        ):
            mock_bus.subscribe.return_value = queue

            task = asyncio.create_task(svc.run())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_to_thread.assert_awaited_once_with(svc.deliver_pending, "abc123", registry=None)
