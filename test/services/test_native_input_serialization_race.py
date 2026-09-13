"""Regression tests for the native-input capture/dispatch race (correction-971).

Background (proven defect): ``terminal_service.send_input`` has four call
sites (the raw ``POST /terminals/{id}/input`` route, ``agent_step.py``'s
initial assignment dispatch, ``InboxService.deliver_pending``'s deferred
delivery, and ``agui/handoff_approval.py``). Before this fix, only
``InboxService`` serialized its OWN sequential deliveries against each other,
via a private per-terminal ``threading.Lock`` -- nothing serialized any of
the four entry points against EACH OTHER for the same ``terminal_id``. Two
``send_input`` calls for the same terminal could therefore have their tmux
bracketed-paste writes genuinely overlap in the pane, merging two logically
distinct native "turns" into one from the provider's point of view. In the
field, this corrupted the dispatch-identity hash that
``bind_completion_dispatch`` records for an assigned worker's first-user
turn: an unrelated follow-up message physically landed inside the still-open
paste of the original assignment, with no turn boundary between them.

Both tests below use a ``threading.Barrier`` to force genuinely simultaneous
contention (not a sleep-timed guess at when a race "might" happen), and a
deliberately widened critical section (a short, bounded sleep *inside* the
mocked tmux write / lock-holding section) purely to make any missing
mutual exclusion easy to catch -- the pass/fail assertion itself is a hard
structural invariant (peak concurrent occupancy of the critical section),
never a timing comparison.

These are written to fail on the pre-fix code (no shared serialization
primitive existed) and pass once ``terminal_service`` guards ``send_input``
with a stable per-terminal lock. See test_terminal_service_full.py for the
general send_input behavioral suite; this file is scoped to the concurrency
regression only.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.models.inbox import OrchestrationType
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import terminal_service
from cli_agent_orchestrator.services.status_monitor import status_monitor


class TestNativeInputNeverOverlapsForSameTerminal:
    """send_input must never let two callers' tmux writes overlap for one terminal."""

    @patch("cli_agent_orchestrator.services.terminal_service.MemoryService")
    @patch("cli_agent_orchestrator.services.terminal_service.status_monitor")
    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_concurrent_send_input_same_terminal_never_overlaps_tmux_write(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
        mock_status_monitor,
        mock_memory_service,
    ):
        """N concurrent send_input() calls for ONE terminal_id: peak concurrent
        occupancy of the tmux-write critical section must be exactly 1.

        A ``threading.Barrier`` releases all callers at once so contention is
        real and simultaneous, not scheduled hopefully. The mocked tmux write
        holds its (would-be) critical section open for a short bounded sleep
        purely to widen the detection window -- with no serialization at all,
        6 threads racing through an unguarded ~20ms window reproduce the
        overlap essentially every run.
        """
        mock_memory_service.return_value.get_curated_memory_context.return_value = ""
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        mock_provider = mock_pm.get_provider.return_value
        mock_provider.paste_enter_count = 2
        mock_provider.paste_submit_delay = 0.0

        occupancy = 0
        max_seen = 0
        occupancy_guard = threading.Lock()  # protects the plain ints below only;
        # this is test-side bookkeeping, not the mechanism under test.

        def send_keys_side_effect(session, window, message, **kwargs):
            nonlocal occupancy, max_seen
            with occupancy_guard:
                occupancy += 1
                max_seen = max(max_seen, occupancy)
            time.sleep(0.02)
            with occupancy_guard:
                occupancy -= 1

        mock_tmux.send_keys.side_effect = send_keys_side_effect

        num_callers = 6
        barrier = threading.Barrier(num_callers)

        def call(i: int) -> None:
            barrier.wait(timeout=5)
            terminal_service.send_input("race-terminal", f"message {i}")

        with ThreadPoolExecutor(max_workers=num_callers) as pool:
            futures = [pool.submit(call, i) for i in range(num_callers)]
            for f in futures:
                f.result(timeout=5)

        assert mock_tmux.send_keys.call_count == num_callers
        assert max_seen == 1, (
            f"native input paste interleaving reproduced -- {max_seen} concurrent "
            "tmux writes were in flight for the same terminal_id at once "
            "(correction-971)"
        )

    @patch("cli_agent_orchestrator.services.terminal_service.MemoryService")
    @patch("cli_agent_orchestrator.services.terminal_service.status_monitor")
    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_concurrent_send_input_different_terminals_still_run_concurrently(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
        mock_status_monitor,
        mock_memory_service,
    ):
        """The fix must be per-terminal, not a global bottleneck: callers for
        DIFFERENT terminal_ids must be able to have their tmux writes overlap.
        """
        mock_memory_service.return_value.get_curated_memory_context.return_value = ""
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        mock_provider = mock_pm.get_provider.return_value
        mock_provider.paste_enter_count = 2
        mock_provider.paste_submit_delay = 0.0

        occupancy = 0
        max_seen = 0
        occupancy_guard = threading.Lock()

        def send_keys_side_effect(session, window, message, **kwargs):
            nonlocal occupancy, max_seen
            with occupancy_guard:
                occupancy += 1
                max_seen = max(max_seen, occupancy)
            time.sleep(0.05)
            with occupancy_guard:
                occupancy -= 1

        mock_tmux.send_keys.side_effect = send_keys_side_effect

        num_callers = 4
        barrier = threading.Barrier(num_callers)

        def call(i: int) -> None:
            barrier.wait(timeout=5)
            terminal_service.send_input(f"distinct-terminal-{i}", "message")

        with ThreadPoolExecutor(max_workers=num_callers) as pool:
            futures = [pool.submit(call, i) for i in range(num_callers)]
            for f in futures:
                f.result(timeout=5)

        assert mock_tmux.send_keys.call_count == num_callers
        assert max_seen > 1, (
            "per-terminal lock appears to be a global bottleneck -- distinct "
            "terminal_ids never ran their tmux writes concurrently"
        )


class TestSendSpecialKeySharesTheSameBoundary:
    """send_special_key (used by agui/handoff_approval.py's C-u clear-before-
    paste) must serialize against send_input for the same terminal too --
    both are genuine tmux writes to the same pane.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.MemoryService")
    @patch("cli_agent_orchestrator.services.terminal_service.status_monitor")
    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_special_key_and_send_input_never_overlap_for_same_terminal(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
        mock_status_monitor,
        mock_memory_service,
    ):
        mock_memory_service.return_value.get_curated_memory_context.return_value = ""
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        mock_provider = mock_pm.get_provider.return_value
        mock_provider.paste_enter_count = 2
        mock_provider.paste_submit_delay = 0.0

        occupancy = 0
        max_seen = 0
        occupancy_guard = threading.Lock()

        def widen(*args, **kwargs):
            nonlocal occupancy, max_seen
            with occupancy_guard:
                occupancy += 1
                max_seen = max(max_seen, occupancy)
            time.sleep(0.02)
            with occupancy_guard:
                occupancy -= 1

        mock_tmux.send_keys.side_effect = widen
        mock_tmux.send_special_key.side_effect = widen

        barrier = threading.Barrier(2)

        def call_send_input():
            barrier.wait(timeout=5)
            terminal_service.send_input("special-key-race-terminal", "hello")

        def call_special_key():
            barrier.wait(timeout=5)
            terminal_service.send_special_key("special-key-race-terminal", "C-u")

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(call_send_input)
            f2 = pool.submit(call_special_key)
            f1.result(timeout=5)
            f2.result(timeout=5)

        assert (
            max_seen == 1
        ), f"send_special_key and send_input tmux writes overlapped ({max_seen} concurrent)"


class TestDispatchIdentityImmutableUnderConcurrentFollowUp:
    """Item B: a concurrent follow-up must never become part of the original
    assignment's registered dispatch identity.

    Uses the REAL StatusMonitor singleton (not mocked) so the native-
    acceptance fence (correction-984, see TestNativeAcceptanceFence below)
    actually engages: the ASSIGN dispatch is sent first and its acceptance is
    explicitly, deliberately confirmed (simulating StatusMonitor's real
    detection loop observing the PROCESSING transition) before the follow-up
    is ever attempted. This is deliberate, not a symmetric race -- unlike a
    Barrier-released simultaneous start (which the mutex alone cannot put in
    a guaranteed order, and previously required sorted() to tolerate either
    winning), this models the actual incident shape: an assignment dispatches
    first, and a follow-up arrives afterward. Proves: (1) bind_completion_
    dispatch is called exactly once, with the assignment's own exact bytes --
    never the follow-up's, and never a concatenation of both; (2) the two
    tmux writes never overlap; (3) they land in EXACT order (assignment,
    then follow-up) with exact, unmodified, Unicode/multiline-preserving
    payloads and exactly one submit each -- never merely "both happened,
    in some order" (sorted()'s weaker guarantee, replaced here per
    correction-984).
    """

    @patch("cli_agent_orchestrator.services.provider_completion_report.bind_completion_dispatch")
    @patch("cli_agent_orchestrator.services.terminal_service.inject_memory_context")
    @patch("cli_agent_orchestrator.services.terminal_service.get_assigned_worker_callback")
    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_assign_then_follow_up_land_in_exact_order_with_identity_bound_only_to_assign(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
        mock_get_callback,
        mock_inject,
        mock_bind,
    ):
        terminal_id = "real-status-terminal-order"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
            "provider": "codex",
        }
        callback = MagicMock()
        callback.completion_id = "0" * 32
        mock_get_callback.return_value = callback
        # inject_memory_context is only meaningful for the ASSIGN call in this
        # test (the follow-up doesn't go through the ASSIGN binding branch at
        # all); pass every message through unchanged so both payloads stay
        # byte-identical to what the caller sent.
        mock_inject.side_effect = lambda message, terminal_id, frozen_memory=None: message
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 2
        provider.paste_submit_delay = 0.0
        # A MagicMock provider's assume_processing_on_dispatch is never `is
        # True`, matching every real provider except the opt-in few -- the
        # native-acceptance fence is meant to engage here.

        writes: list[str] = []

        def send_keys_side_effect(session, window, message, **kwargs):
            writes.append(message)

        mock_tmux.send_keys.side_effect = send_keys_side_effect

        # Unicode + multiline, to prove the fence and the identity binding
        # both preserve exact bytes rather than merely "some text".
        assign_text = "assigned task: über-review — line one\nline two"
        follow_up_text = "message874 follow-up: 日本語 — line one\nline two"

        try:
            terminal_service.send_input(
                terminal_id, assign_text, orchestration_type=OrchestrationType.ASSIGN
            )
            assert writes == [assign_text], "assignment must paste before anything else"

            # Simulate StatusMonitor's real detection loop observing the
            # dispatch's acceptance (a genuine PROCESSING transition) --
            # deliberately, only now, releasing the fence armed by the
            # ASSIGN call's own notify_input_sent.
            status_monitor._apply_detection(terminal_id, TerminalStatus.PROCESSING)

            terminal_service.send_input(
                terminal_id, follow_up_text, orchestration_type=OrchestrationType.SEND_MESSAGE
            )
        finally:
            status_monitor.clear_terminal(terminal_id)

        assert writes == [
            assign_text,
            follow_up_text,
        ], f"expected exact order [assign, follow-up], got {writes!r}"
        assert mock_tmux.send_keys.call_count == 2, "exactly one submit per input"
        mock_bind.assert_called_once_with("codex", terminal_id, "0" * 32, assign_text)


class TestNativeAcceptanceFence:
    """Item A: a per-terminal mutex alone (correction-971) is not enough --
    a SUBSEQUENT send_input() call must also wait for a PRIOR dispatch's
    native acceptance (correction-984) before it may paste, and must fail
    closed rather than wait forever or silently proceed.

    Uses the REAL StatusMonitor singleton so the fence armed by
    notify_input_sent() and released by _apply_detection() actually engages;
    only the tmux backend, terminal metadata, and provider manager are
    mocked.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_follow_up_blocks_until_a_controllably_delayed_acceptance_is_observed(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        terminal_id = "fence-terminal-delayed-acceptance"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        writes: list[str] = []

        def send_keys_side_effect(session, window, message, **kwargs):
            writes.append(message)

        mock_tmux.send_keys.side_effect = send_keys_side_effect

        first_text = "initial assignment dispatch"
        second_text = "message874 follow-up"

        try:
            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]

            # Acceptance is deliberately NOT yet observed: the fence armed by
            # the call above is still pending. Start the follow-up on its own
            # thread -- it must block inside wait_for_native_acceptance
            # rather than paste immediately.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(terminal_service.send_input, terminal_id, second_text)

                # Not the correctness mechanism (that is the fence itself) --
                # a bounded scheduling nudge so the assertion below is
                # actually exercising "still blocked", not "hasn't started
                # yet". A blocked thread.Event.wait() releases the GIL almost
                # immediately, so this is far more time than needed in
                # practice; the real proof is what follows the release.
                time.sleep(0.2)
                assert writes == [first_text], (
                    "follow-up pasted before the controlled acceptance event fired -- "
                    "the native-acceptance fence did not hold"
                )

                # NOW deliberately deliver the controlled acceptance event.
                status_monitor._apply_detection(terminal_id, TerminalStatus.PROCESSING)
                future.result(timeout=5)

            assert writes == [
                first_text,
                second_text,
            ], f"expected exact order [first, follow-up] once accepted, got {writes!r}"
        finally:
            status_monitor.clear_terminal(terminal_id)

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_follow_up_fails_closed_when_acceptance_never_arrives(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
        monkeypatch,
    ):
        terminal_id = "fence-terminal-never-accepted"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0
        mock_tmux.send_keys.side_effect = lambda *a, **k: None
        # Bounded, and short -- this test's whole point is proving the
        # timeout path itself, not how long it takes.
        monkeypatch.setattr(terminal_service, "NATIVE_ACCEPTANCE_TIMEOUT_S", 0.2)

        try:
            terminal_service.send_input(terminal_id, "initial assignment dispatch")
            assert mock_tmux.send_keys.call_count == 1

            # Acceptance is never delivered for this terminal_id in this test.
            with pytest.raises(terminal_service.NativeAcceptanceTimeoutError):
                terminal_service.send_input(terminal_id, "message874 follow-up")

            assert (
                mock_tmux.send_keys.call_count == 1
            ), "the follow-up must never physically paste when acceptance times out"
        finally:
            status_monitor.clear_terminal(terminal_id)


class TestTerminalInputLockPrimitive:
    """Direct, pure event/barrier tests of the serialization primitive itself."""

    def test_lock_is_stable_per_terminal_and_distinct_across_terminals(self):
        lock_fn = getattr(terminal_service, "terminal_input_lock", None)
        if lock_fn is None:
            pytest.fail(
                "terminal_service.terminal_input_lock is missing -- the shared "
                "per-terminal native-input serialization primitive for "
                "correction-971 has not been added"
            )
        a1 = lock_fn("lock-terminal-a")
        a2 = lock_fn("lock-terminal-a")
        b1 = lock_fn("lock-terminal-b")
        assert a1 is a2, "terminal_input_lock must return the SAME lock object per terminal_id"
        assert a1 is not b1, "terminal_input_lock must return DIFFERENT locks for different ids"

    def test_lock_is_reentrant_on_the_same_thread(self):
        """InboxService needs to hold this lock across its own pre-send check
        (wait_for_capture_before_input) and then into send_input itself,
        which re-acquires the same lock internally -- same-thread reentrant
        acquisition must not deadlock.
        """
        lock_fn = getattr(terminal_service, "terminal_input_lock", None)
        if lock_fn is None:
            pytest.fail(
                "terminal_service.terminal_input_lock is missing -- the shared "
                "per-terminal native-input serialization primitive for "
                "correction-971 has not been added"
            )
        lock = lock_fn("reentrant-terminal")
        acquired_twice = False
        with lock, lock:
            acquired_twice = True
        assert acquired_twice

    def test_lock_never_allows_two_concurrent_holders_for_same_terminal(self):
        lock_fn = getattr(terminal_service, "terminal_input_lock", None)
        if lock_fn is None:
            pytest.fail(
                "terminal_service.terminal_input_lock is missing -- the shared "
                "per-terminal native-input serialization primitive for "
                "correction-971 has not been added"
            )
        lock = lock_fn("contended-terminal")
        occupancy = 0
        max_seen = 0
        occupancy_guard = threading.Lock()
        num_workers = 8
        barrier = threading.Barrier(num_workers)

        def worker():
            nonlocal occupancy, max_seen
            barrier.wait(timeout=5)
            with lock:
                with occupancy_guard:
                    occupancy += 1
                    max_seen = max(max_seen, occupancy)
                time.sleep(0.01)
                with occupancy_guard:
                    occupancy -= 1

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker) for _ in range(num_workers)]
            for f in futures:
                f.result(timeout=5)

        assert max_seen == 1
