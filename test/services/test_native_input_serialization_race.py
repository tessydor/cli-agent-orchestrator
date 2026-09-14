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

# send_input() ultimately reaches inject_memory_context/MemoryService, which
# touches the database. Most tests below mock that path away, but a few
# (e.g. TestCompletionCaptureRecheckedAfterAcceptanceWait) exercise real
# terminal_service/status_monitor/assigned_worker_completion_service code
# without mocking it. Without this isolated per-test DB, whatever
# db.SessionLocal happens to be pointed at when this file runs (which varies
# by what other test files/fixtures ran earlier in the same session) leaks
# in -- matching the same isolation test_terminal_service_full.py already
# requires for the identical reason.
pytestmark = pytest.mark.usefixtures("isolated_memory_db")


def _apply_detection_with_fresh_evidence(terminal_id: str, detected: TerminalStatus) -> None:
    """Clearly-scoped test helper (correction-1001): tests that simulate the
    real chunk-driven detection loop by calling status_monitor.
    _apply_detection() directly bypass _process_chunk entirely, so they
    never stamp _buffer_changed_at the way a genuine chunk arrival would --
    and since correction-1001, a missing freshness timestamp now fails
    CLOSED (same as UNKNOWN), exactly like real production code. Tests that
    mean to simulate GENUINE post-arm acceptance evidence must supply that
    freshness explicitly, here, rather than relying on any implicit
    fail-open default that no longer exists.
    """
    status_monitor._buffer_changed_at[terminal_id] = time.monotonic()
    status_monitor._apply_detection(terminal_id, detected)


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
            _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.PROCESSING)

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
                _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.PROCESSING)
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
    def test_unknown_before_acceptance_never_releases_the_fence(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        """Correction-994: UNKNOWN is "no signal", never acceptance evidence.

        A brand-new terminal_id has no prior _last_status, so UNKNOWN passes
        the early "UNKNOWN never overwrites a known status" guard (last is
        None) and, before this fix, reached the release code and
        incorrectly released the fence on the very first post-dispatch
        detection -- exactly the scenario a fresh ASSIGN dispatch to a new
        terminal can hit before its pane stabilizes.
        """
        terminal_id = "fence-terminal-unknown-before-acceptance"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        writes: list[str] = []
        mock_tmux.send_keys.side_effect = lambda session, window, message, **kw: writes.append(
            message
        )

        first_text = "initial assignment dispatch"
        second_text = "message874 follow-up"

        try:
            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]

            # UNKNOWN must never count as acceptance -- the fence stays armed.
            status_monitor._apply_detection(terminal_id, TerminalStatus.UNKNOWN)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(terminal_service.send_input, terminal_id, second_text)
                time.sleep(0.2)
                assert writes == [first_text], (
                    "follow-up pasted after only an UNKNOWN detection -- UNKNOWN must "
                    "never release the native-acceptance fence"
                )

                # A genuine post-arm transition still releases it correctly.
                _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.PROCESSING)
                future.result(timeout=5)

            assert writes == [first_text, second_text]
        finally:
            status_monitor.clear_terminal(terminal_id)

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_stale_pre_arm_evidence_never_releases_but_fresh_evidence_does(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        """Correction-994: evidence timestamped BEFORE this fence's own arm
        (a chunk that was already in flight, whose _process_chunk handling
        was merely delayed past arm by independent thread/asyncio
        scheduling -- not reflecting this dispatch at all) must not release
        it. Genuinely fresh evidence, timestamped at/after arm, still does.
        """
        terminal_id = "fence-terminal-stale-pre-arm-evidence"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        writes: list[str] = []
        mock_tmux.send_keys.side_effect = lambda session, window, message, **kw: writes.append(
            message
        )

        first_text = "initial assignment dispatch"
        second_text = "message874 follow-up"

        try:
            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]

            # Simulate a chunk that was ALREADY captured before this dispatch
            # (its timestamp predates arm), whose _apply_detection call is
            # only reaching us now -- PROCESSING chosen deliberately: it
            # does not participate in the sticky-ready suppress guards, so
            # this test isolates the acceptance-fence timestamp check from
            # that unrelated, pre-existing mechanism.
            status_monitor._buffer_changed_at[terminal_id] = time.monotonic() - 100.0
            status_monitor._apply_detection(terminal_id, TerminalStatus.PROCESSING)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(terminal_service.send_input, terminal_id, second_text)
                time.sleep(0.2)
                assert writes == [first_text], (
                    "follow-up pasted after only stale, pre-arm-timestamped evidence -- "
                    "the native-acceptance fence did not hold"
                )

                # Genuinely fresh (post-arm) evidence still releases it.
                status_monitor._buffer_changed_at[terminal_id] = time.monotonic()
                status_monitor._apply_detection(terminal_id, TerminalStatus.COMPLETED)
                future.result(timeout=5)

            assert writes == [first_text, second_text]
        finally:
            status_monitor.clear_terminal(terminal_id)

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_missing_evidence_timestamp_never_releases_but_explicit_evidence_does(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        """Correction-1001 item 1: a non-IDLE detection with NO freshness
        evidence at all (no _buffer_changed_at entry -- distinct from the
        UNKNOWN case above, and from the deliberately-STALE-timestamped case
        below) must fail CLOSED exactly like UNKNOWN, not fail open. Only an
        explicit, fresh post-arm timestamp may release it.
        """
        terminal_id = "fence-terminal-missing-evidence"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        writes: list[str] = []
        mock_tmux.send_keys.side_effect = lambda session, window, message, **kw: writes.append(
            message
        )

        first_text = "initial assignment dispatch"
        second_text = "message874 follow-up"

        try:
            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]
            # No _buffer_changed_at entry exists for this terminal_id at
            # all -- this is the "missing", not merely "stale", case.
            assert terminal_id not in status_monitor._buffer_changed_at

            status_monitor._apply_detection(terminal_id, TerminalStatus.PROCESSING)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(terminal_service.send_input, terminal_id, second_text)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.5)
                assert writes == [first_text], (
                    "follow-up pasted after a non-IDLE detection with NO freshness evidence -- "
                    "missing evidence must fail closed, not fail open"
                )

                # Explicit, fresh post-arm evidence still releases it.
                _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.COMPLETED)
                future.result(timeout=5)

            assert writes == [first_text, second_text]
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


class TestNonSubmittingSpecialKeysDoNotFenceTheNextTurn:
    """Correction-994 item A: a special key that never causes a native
    processing transition (C-u clearing the composer; menu Up/Down
    navigation) must not arm or wait on the acceptance fence a REAL
    submitted turn checks -- otherwise the very next genuine send blocks for
    the full NATIVE_ACCEPTANCE_TIMEOUT_S and then fails closed for no
    reason. Uses the REAL StatusMonitor singleton so the fence actually
    engages; only the tmux backend/metadata/provider are mocked.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_handoff_approval_c_u_then_send_input_completes_without_delay(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        from cli_agent_orchestrator.services.agui.handoff_approval import (
            TerminalServiceAnswerDelivery,
        )

        terminal_id = "handoff-c-u-terminal"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        calls: list[tuple] = []
        mock_tmux.send_special_key.side_effect = lambda *a, **k: calls.append(("key", a, k))
        mock_tmux.send_keys.side_effect = lambda *a, **k: calls.append(("input", a, k))

        try:
            started = time.monotonic()
            TerminalServiceAnswerDelivery().send_input(terminal_id, "the approval answer")
            elapsed = time.monotonic() - started
        finally:
            status_monitor.clear_terminal(terminal_id)

        assert elapsed < 1.0, (
            f"C-u incorrectly armed/waited on the acceptance fence -- took {elapsed:.2f}s "
            "(expected well under NATIVE_ACCEPTANCE_TIMEOUT_S)"
        )
        assert [c[0] for c in calls] == ["key", "input"], "exactly one clear, one text submit"

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_menu_navigation_does_not_fence_the_answering_enter(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        terminal_id = "menu-nav-terminal"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        mock_tmux.send_special_key.side_effect = lambda *a, **k: None

        try:
            started = time.monotonic()
            terminal_service.send_special_key(terminal_id, "Down", submits_turn=False)
            terminal_service.send_special_key(terminal_id, "Down", submits_turn=False)
            terminal_service.send_special_key(terminal_id, "Enter")  # the real answer, submits
            elapsed = time.monotonic() - started
        finally:
            status_monitor.clear_terminal(terminal_id)

        assert elapsed < 1.0, (
            f"non-submitting navigation incorrectly fenced the answering Enter -- "
            f"took {elapsed:.2f}s"
        )
        assert mock_tmux.send_special_key.call_count == 3


class TestPluginDispatchRunsOutsideTheTerminalInputLock:
    """Correction-994 item 3: dispatch_plugin_event can run a plugin's
    handler SYNCHRONOUSLY (asyncio.run) when no event loop is active on the
    calling thread -- a genuine, untrusted, out-of-tree code execution that
    must never happen while the per-terminal physical-write/acceptance-fence
    lock is held, since any plugin that (directly or transitively) touches
    the same terminal_id from a different thread would otherwise deadlock
    against the still-blocked original call.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_plugin_dispatch_finds_the_lock_already_released(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_update,
    ):
        terminal_id = "plugin-dispatch-terminal"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        events: list[tuple] = []
        mock_tmux.send_keys.side_effect = lambda *a, **k: events.append(("sent",))

        class _FakeRegistry:
            async def dispatch(self, event_type, event):
                # dispatch_plugin_event's no-loop branch runs THIS coroutine
                # synchronously (asyncio.run) on send_input()'s OWN thread --
                # exactly the case that used to execute while send_input()'s
                # `with terminal_input_lock(...)` was still open. Checking
                # the lock from THIS same thread would be meaningless
                # (RLock's same-thread reentrance would report "free" either
                # way); probe from a genuinely SEPARATE thread instead, whose
                # acquire attempt only succeeds if NO thread -- including
                # this one -- currently holds it.
                lock = terminal_service.terminal_input_lock(terminal_id)
                with ThreadPoolExecutor(max_workers=1) as probe_pool:
                    acquired = probe_pool.submit(lock.acquire, True, 1.0).result(timeout=5)
                events.append(("dispatch", event_type, acquired))
                if acquired:
                    lock.release()

        try:
            result = terminal_service.send_input(
                terminal_id,
                "hello",
                registry=_FakeRegistry(),
                sender_id="caller-1",
                orchestration_type=OrchestrationType.SEND_MESSAGE,
            )
        finally:
            status_monitor.clear_terminal(terminal_id)

        assert result is True
        assert events == [
            ("sent",),
            ("dispatch", "post_send_message", True),
        ], f"expected exactly one dispatch, after the send, with the lock free; got {events!r}"


class TestCompletionCaptureRecheckedAfterAcceptanceWait:
    """Correction-994 item B: send_input() must recheck BOTH native
    acceptance AND completion-capture eligibility at its one shared
    boundary. A caller (e.g. InboxService) can legitimately read IDLE and
    skip its own COMPLETED-capture check, only for the terminal to finish
    its turn WHILE this call is blocked inside the acceptance wait -- the
    SAME detection event that releases the acceptance fence also arms
    AssignedWorkerCompletionService's capture barrier. Uses the REAL
    StatusMonitor AND the REAL AssignedWorkerCompletionService singletons so
    both barriers actually engage; only the tmux backend/metadata/provider
    are mocked.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.status_monitor.provider_manager")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_fast_completion_during_acceptance_wait_still_blocks_on_capture(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_status_pm,
        mock_update,
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        # status_monitor.py imports provider_manager independently from
        # terminal_service.py (a SEPARATE module-level name binding, even
        # though both normally point at the same real singleton) -- mocking
        # only terminal_service's copy leaves status_monitor.get_status()'s
        # own get_backend().supports_event_inbox() branch calling into the
        # REAL provider manager. None here forces that branch to fall
        # through to the real _last_status-based cached logic this test
        # actually needs, deterministically, regardless of what the real
        # provider manager happens to have cached from other tests.
        mock_status_pm.get_provider.return_value = None

        terminal_id = "completion-recheck-terminal"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0

        writes: list[str] = []
        mock_tmux.send_keys.side_effect = lambda session, window, message, **kw: writes.append(
            message
        )

        first_text = "initial assignment dispatch"
        second_text = "message874 follow-up"

        try:
            # A known assigned worker: announce_terminal_status only arms a
            # capture barrier for terminals register_assignment has marked.
            assigned_worker_completion_service.register_assignment(terminal_id)

            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]

            with ThreadPoolExecutor(max_workers=1) as pool:
                # Blocks inside wait_for_native_acceptance -- nothing has
                # released the fence armed by the dispatch above yet.
                # future.result(timeout=...) raising TimeoutError is the
                # synchronization-based proof of "still genuinely blocked"
                # (it can only return once the call actually finishes,
                # success or exception) -- robust under system load, unlike
                # a fixed sleep-then-check-a-side-effect assertion.
                future = pool.submit(terminal_service.send_input, terminal_id, second_text)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.5)
                assert writes == [first_text]

                # The first turn completes WHILE the follow-up is still
                # blocked on acceptance. The same detection event releases
                # the acceptance fence (COMPLETED != IDLE) AND arms the
                # capture barrier (announce_terminal_status, called
                # synchronously from inside _apply_detection_locked).
                _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.COMPLETED)

                # Acceptance is now satisfied, but the capture barrier is
                # STILL armed (nothing has released it yet) -- the
                # follow-up must remain blocked, now on the FRESH capture
                # recheck, not the (already-cleared) acceptance fence.
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.5)
                assert writes == [first_text], (
                    "follow-up pasted into a COMPLETED terminal before its report was "
                    "durably captured -- correction-994's capture recheck did not hold"
                )

                # NOW the capture completes.
                assigned_worker_completion_service._release_capture_barrier(terminal_id)
                future.result(timeout=5)

            assert writes == [
                first_text,
                second_text,
            ], f"expected exact order [first, follow-up] once captured, got {writes!r}"
        finally:
            status_monitor.clear_terminal(terminal_id)
            assigned_worker_completion_service._release_capture_barrier(terminal_id)


class TestTwoContendersGuardSequenceIsCoherentAcrossACaptureRelease:
    """Correction-1017: the Condition-based capture wait (correction-1007)
    genuinely relinquishes terminal_input_lock while blocked, so a SECOND
    concurrent send_input() contender for the SAME terminal can run its
    OWN acceptance/status checks -- and enter the SAME capture wait --
    while the FIRST contender is still parked there. One capture release
    wakes both; only ONE may physically submit before a fresh native-
    acceptance event confirms that submission. The existing one-waiter
    test above (TestCompletionCaptureRecheckedAfterAcceptanceWait) cannot
    see this: it never has two contenders both actually block on the same
    barrier. Uses only real threading.Event/Barrier synchronization and
    bounded future.result() timeouts -- no sleeps as the proof.
    """

    @patch("cli_agent_orchestrator.services.terminal_service.update_last_active")
    @patch("cli_agent_orchestrator.services.status_monitor.provider_manager")
    @patch("cli_agent_orchestrator.services.terminal_service.provider_manager")
    @patch("cli_agent_orchestrator.backends.registry._backend")
    @patch("cli_agent_orchestrator.services.terminal_service.get_terminal_metadata")
    def test_two_contenders_one_release_one_submit_then_fresh_acceptance_then_the_other(
        self,
        mock_get_metadata,
        mock_tmux,
        mock_pm,
        mock_status_pm,
        mock_update,
    ):
        from cli_agent_orchestrator.services.assigned_worker_completion_service import (
            assigned_worker_completion_service,
        )

        # Same rationale as the sibling one-waiter test above: status_monitor.py
        # imports provider_manager independently, so its own event-inbox branch
        # must be forced off deterministically.
        mock_status_pm.get_provider.return_value = None

        terminal_id = "two-contender-terminal"
        mock_get_metadata.return_value = {
            "tmux_session": "cao-session",
            "tmux_window": "developer-abcd",
        }
        provider = mock_pm.get_provider.return_value
        provider.paste_enter_count = 1
        provider.paste_submit_delay = 0.0
        provider.assume_processing_on_dispatch = False

        writes: list[str] = []
        mock_tmux.send_keys.side_effect = lambda session, window, message, **kw: writes.append(
            message
        )

        first_text = "initial assignment dispatch"
        contender_a_text = "contender A queued follow-up"
        contender_b_text = "contender B queued follow-up"

        # Instrument the REAL _wait_for_capture_before_input_detailed to signal
        # when each of the first two callers is about to enter the real
        # Condition.wait() -- the deterministic "both contenders are now
        # genuinely parked on the same barrier" proof this test needs, without
        # ever touching the underlying wait/notify logic itself.
        real_detailed_wait = (
            assigned_worker_completion_service._wait_for_capture_before_input_detailed
        )
        entered_wait = [threading.Event(), threading.Event()]
        call_order: list[str] = []
        call_order_lock = threading.Lock()

        def instrumented_detailed_wait(worker_terminal_id, timeout=5.0):
            with call_order_lock:
                index = len(call_order)
                call_order.append(worker_terminal_id)
            if index < 2:
                entered_wait[index].set()
            return real_detailed_wait(worker_terminal_id, timeout)

        try:
            assigned_worker_completion_service.register_assignment(terminal_id)

            terminal_service.send_input(terminal_id, first_text)
            assert writes == [first_text]

            # Complete the first turn: releases the acceptance fence AND
            # arms the capture barrier (announce_terminal_status, called
            # synchronously from inside _apply_detection_locked) for this
            # known/registered worker.
            _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.COMPLETED)

            with patch.object(
                assigned_worker_completion_service,
                "_wait_for_capture_before_input_detailed",
                instrumented_detailed_wait,
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_a = pool.submit(
                        terminal_service.send_input, terminal_id, contender_a_text
                    )
                    future_b = pool.submit(
                        terminal_service.send_input, terminal_id, contender_b_text
                    )

                    # Both contenders must ACTUALLY enter the real capture
                    # wait -- i.e. both genuinely relinquished
                    # terminal_input_lock via Condition.wait() -- before
                    # this test releases the barrier. This is the concrete
                    # two-waiter interleaving 1017 requires; the lock
                    # itself already serializes each contender's own
                    # acceptance+status read strictly before it can reach
                    # this point, so waiting for both signals is sufficient
                    # (no sleep-based timing assumption).
                    assert entered_wait[0].wait(
                        timeout=5
                    ), "contender 1 never reached the capture wait"
                    assert entered_wait[1].wait(
                        timeout=5
                    ), "contender 2 never reached the capture wait"
                    assert writes == [first_text], "no paste may occur while both are still waiting"

                    # One capture release wakes both waiters.
                    assigned_worker_completion_service._release_capture_barrier(terminal_id)

                    # Exactly one physical submit occurs after the release
                    # -- the other contender must still be blocked, now on
                    # native acceptance for the winner's brand-new,
                    # deliberately-unconfirmed dispatch fence.
                    winner_future = None
                    for candidate in (future_a, future_b):
                        try:
                            candidate.result(timeout=3)
                            winner_future = candidate
                        except TimeoutError:
                            continue
                    assert (
                        winner_future is not None
                    ), "neither contender submitted after the release"
                    loser_future = future_b if winner_future is future_a else future_a
                    assert len(writes) == 2, f"expected exactly one new submit, got {writes!r}"
                    winner_text = writes[1]
                    assert winner_text in (contender_a_text, contender_b_text)

                    # The loser must still be blocked -- proving it did NOT
                    # act on its stale pre-wait acceptance/status snapshot.
                    with pytest.raises(TimeoutError):
                        loser_future.result(timeout=0.5)
                    assert writes == [first_text, winner_text], (
                        "the second contender pasted before a fresh acceptance event for "
                        "the winner's own new dispatch -- correction-1017's stale-guard race"
                    )

                    # A fresh native-acceptance event for the winner's own
                    # dispatch (a real detection, standing in for the real
                    # StatusMonitor loop) is the ONLY thing that may release
                    # the loser now.
                    _apply_detection_with_fresh_evidence(terminal_id, TerminalStatus.PROCESSING)
                    loser_future.result(timeout=5)

            loser_text = contender_b_text if winner_text == contender_a_text else contender_a_text
            assert writes == [
                first_text,
                winner_text,
                loser_text,
            ], f"expected exactly one submit per contender, in order, got {writes!r}"
        finally:
            status_monitor.clear_terminal(terminal_id)
            assigned_worker_completion_service._release_capture_barrier(terminal_id)


class TestDrainAllTerminalInputLocks:
    """Correction-1038 Part C.3: shutdown must be able to prove no physical
    native-input write is currently in flight for ANY known terminal,
    before the process is replaced. Deliberately reuses the exact
    ``terminal_input_lock`` every ``send_input`` entry point already
    serializes through -- proven here directly with real threads holding
    (and later releasing) that real lock, no mocking of the lock itself.
    """

    def test_a_free_lock_drains_immediately(self):
        terminal_id = "drain-free-terminal"
        # Merely calling terminal_input_lock() registers it in the module's
        # dict; this test never acquires it, so it must read as free.
        terminal_service.terminal_input_lock(terminal_id)
        try:
            still_busy = terminal_service.drain_all_terminal_input_locks(timeout=1.0)
            assert terminal_id not in still_busy
        finally:
            with terminal_service._terminal_input_locks_guard:
                terminal_service._terminal_input_locks.pop(terminal_id, None)

    def test_a_held_lock_is_reported_busy_then_drains_after_release(self):
        terminal_id = "drain-held-terminal"
        lock = terminal_service.terminal_input_lock(terminal_id)
        released = threading.Event()
        holder_acquired = threading.Event()

        def hold_then_release():
            with lock:
                holder_acquired.set()
                released.wait(timeout=5)

        holder = threading.Thread(target=hold_then_release)
        holder.start()
        try:
            assert holder_acquired.wait(timeout=5), "holder thread never acquired the lock"

            # Bounded and short: must report busy, not hang until the holder
            # releases on its own.
            still_busy = terminal_service.drain_all_terminal_input_locks(timeout=0.2)
            assert terminal_id in still_busy

            released.set()
            holder.join(timeout=5)
            assert not holder.is_alive()

            # Now genuinely free -- the same check must confirm it.
            still_busy_after = terminal_service.drain_all_terminal_input_locks(timeout=1.0)
            assert terminal_id not in still_busy_after
        finally:
            released.set()
            holder.join(timeout=5)
            with terminal_service._terminal_input_locks_guard:
                terminal_service._terminal_input_locks.pop(terminal_id, None)

    def test_one_stuck_terminal_does_not_starve_the_check_for_others(self):
        stuck_id = "drain-stuck-terminal"
        free_id = "drain-other-free-terminal"
        stuck_lock = terminal_service.terminal_input_lock(stuck_id)
        terminal_service.terminal_input_lock(free_id)
        released = threading.Event()
        holder_acquired = threading.Event()

        def hold_forever_ish():
            with stuck_lock:
                holder_acquired.set()
                released.wait(timeout=5)

        holder = threading.Thread(target=hold_forever_ish)
        holder.start()
        try:
            assert holder_acquired.wait(timeout=5)
            still_busy = terminal_service.drain_all_terminal_input_locks(timeout=0.3)
            assert stuck_id in still_busy
            assert free_id not in still_busy
        finally:
            released.set()
            holder.join(timeout=5)
            with terminal_service._terminal_input_locks_guard:
                terminal_service._terminal_input_locks.pop(stuck_id, None)
                terminal_service._terminal_input_locks.pop(free_id, None)
