"""HerdrInboxService — socket event-based inbox delivery for herdr backend.

Replaces the pipe-pane + file watchdog approach with herdr's native socket API.
Subscribes to a broadcast pane.updated event (whose payload carries
agent_status) and delivers pending inbox messages when a pane transitions to
idle or done.

Design:
- Maintains a pane_id → terminal_id map for managed panes
- Subscribes once to a broadcast pane.updated (no pane_id) covering all panes,
  so a newly registered pane's events already arrive — registration updates the
  map only and never re-subscribes or forces a reconnect
- Reconnects with exponential backoff on socket disconnect
- Supplements with periodic pane read for kiro-cli (working >30s check)
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Exponential backoff parameters
_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_MAX = 30.0  # seconds
_BACKOFF_MULTIPLIER = 2.0

# Kiro supplement check: how long in "working" before we check pane read
_KIRO_WORKING_THRESHOLD = 30.0  # seconds


class HerdrInboxService:
    """Event-driven inbox delivery service using herdr socket API.

    Subscribes to agent status events for managed panes and delivers
    pending messages when agents become idle/done.
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        delivery_callback: Optional[Callable[[str], None]] = None,
        herdr_session: str = "cao",
    ) -> None:
        """Initialize the inbox service.

        Args:
            socket_path: Path to herdr socket. None = auto-detect from env.
            delivery_callback: Function to call for message delivery.
                Signature: callback(terminal_id) → checks and delivers pending messages.
            herdr_session: Name of the herdr session to connect to. Used to
                derive the default socket path and prefix CLI calls.
        """
        self._herdr_session = herdr_session
        self._socket_path = socket_path or self._default_socket_path(herdr_session)
        self._delivery_callback = delivery_callback

        # Managed pane tracking
        self._pane_to_terminal: Dict[str, str] = {}  # pane_id → terminal_id
        self._terminal_to_pane: Dict[str, str] = {}  # terminal_id → pane_id

        # Kiro-specific tracking for supplement check
        self._kiro_terminals: Set[str] = set()  # terminal_ids using kiro-cli
        self._working_since: Dict[str, float] = {}  # terminal_id → timestamp

        # Workspace tracking for lifecycle events
        self._workspace_to_session: Dict[str, str] = {}  # workspace_id → session_name

        # Connection state
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._backoff = _BACKOFF_BASE

    @staticmethod
    def _default_socket_path(session_name: str = "cao") -> str:
        """Determine default herdr socket path for a named session.

        The default session (name ``"default"``) uses a flat path:
        ``~/.config/herdr/herdr.sock``.

        Named sessions use a sessions subdirectory:
        ``~/.config/herdr/sessions/<session_name>/herdr.sock``.

        Args:
            session_name: Herdr session name. Defaults to ``"cao"``.
        """
        import os
        from pathlib import Path

        # Check XDG_CONFIG_HOME first, fallback to ~/.config
        config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        if session_name == "default":
            return f"{config_home}/herdr/herdr.sock"
        return f"{config_home}/herdr/sessions/{session_name}/herdr.sock"

    def register_terminal(self, terminal_id: str, pane_id: str, is_kiro: bool = False) -> None:
        """Register a terminal for event-based inbox delivery.

        Args:
            terminal_id: CAO terminal identifier
            pane_id: Current herdr compact pane_id
            is_kiro: Whether this terminal runs kiro-cli (enables supplement check)
        """
        self._pane_to_terminal[pane_id] = terminal_id
        self._terminal_to_pane[terminal_id] = pane_id
        if is_kiro:
            self._kiro_terminals.add(terminal_id)

        logger.info(f"Registered terminal {terminal_id} (pane={pane_id}, kiro={is_kiro})")

    def unregister_terminal(self, terminal_id: str) -> None:
        """Remove a terminal from managed set.

        Args:
            terminal_id: Terminal to unregister
        """
        pane_id = self._terminal_to_pane.pop(terminal_id, None)
        if pane_id:
            self._pane_to_terminal.pop(pane_id, None)
        self._kiro_terminals.discard(terminal_id)
        self._working_since.pop(terminal_id, None)
        logger.info(f"Unregistered terminal {terminal_id}")

    async def start(self) -> None:
        """Start the event loop: wait for first terminal, then connect and listen."""
        # Run DB cleanup before starting the socket loop so ghost records from
        # prior server runs are removed even when no terminals are registered yet.
        await self._startup_db_cleanup()
        kiro_task = asyncio.ensure_future(self._kiro_supplement_loop())
        try:
            await self._socket_loop()
        finally:
            kiro_task.cancel()

    async def _startup_db_cleanup(self) -> None:
        """Delete ghost DB terminals whose herdr tabs no longer exist.

        Runs once at server startup before any pane registrations.  Cannot
        rely on _pane_to_terminal (empty at startup) or _workspace_to_session
        (populated later by _reconcile).  Builds the workspace map directly
        from a herdr api snapshot.
        """
        from cli_agent_orchestrator.clients.database import list_terminals_by_session
        from cli_agent_orchestrator.services.terminal_service import delete_missing_terminal

        snapshot = self._fetch_snapshot()
        if snapshot is None or not self._snapshot_has_complete_identity(snapshot):
            logger.debug("Startup DB cleanup: no snapshot, skipping")
            return

        # workspace_id -> label (= CAO session name). Skip malformed records.
        workspace_to_session = {
            ws["workspace_id"]: ws["label"]
            for ws in snapshot.get("workspaces", [])
            if ws.get("workspace_id") and ws.get("label")
        }

        # workspace_id -> set of live tab labels (= CAO window names)
        live_tabs_by_workspace: Dict[str, set] = {}
        for tab in snapshot.get("tabs", []):
            ws_id = tab.get("workspace_id", "")
            label = tab.get("label", "")
            if ws_id and label:
                live_tabs_by_workspace.setdefault(ws_id, set()).add(label)

        deleted = 0
        for ws_id, session_name in workspace_to_session.items():
            live_labels = live_tabs_by_workspace.get(ws_id, set())
            db_terminals = list_terminals_by_session(session_name)
            for term in db_terminals:
                window = term.get("tmux_window", "")
                if window and window not in live_labels:
                    logger.info(
                        f"Startup DB cleanup: deleting ghost terminal {term['id']} "
                        f"({session_name}:{window}) — tab not in herdr"
                    )
                    try:
                        if delete_missing_terminal(
                            term["id"],
                            "Herdr startup snapshot proved the persisted worker tab is missing",
                        ):
                            deleted += 1
                    except Exception as e:
                        logger.warning(
                            f"Startup DB cleanup: failed to delete ghost terminal "
                            f"{term['id']}: {e}"
                        )

        if deleted:
            logger.info(f"Startup DB cleanup: removed {deleted} ghost terminal(s)")
        else:
            logger.debug("Startup DB cleanup: no ghost terminals found")

    async def _kiro_supplement_loop(self) -> None:
        """Periodically check kiro terminals stuck in working state."""
        while True:
            await asyncio.sleep(10.0)
            try:
                await self.check_kiro_supplements()
            except Exception:
                logger.debug("Kiro supplement check error", exc_info=True)

    async def _socket_loop(self) -> None:
        """Connect to herdr socket and listen for events with reconnect.

        Defers connection until at least one terminal is registered. This avoids
        the disconnect/reconnect churn caused by herdr closing idle connections
        that have no active subscriptions.
        """
        while True:
            # Wait until there is at least one pane to subscribe to
            while not self._pane_to_terminal:
                await asyncio.sleep(0.5)

            try:
                await self._connect()

                # Reconcile map against live herdr state before subscribing
                await self._reconcile()

                # Subscribe to everything in ONE events.subscribe call: a single
                # broadcast pane.updated (no pane_id) plus the lifecycle events.
                # herdr resets the connection on a second events.subscribe, so
                # this must be a single combined call.
                await self._subscribe_all_events()

                self._backoff = _BACKOFF_BASE  # Reset backoff after successful setup

                # Listen for events
                await self._event_loop()

            except (ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                logger.warning(f"Herdr socket disconnected: {e}")

                # Exponential backoff
                logger.info(f"Reconnecting in {self._backoff}s...")
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * _BACKOFF_MULTIPLIER, _BACKOFF_MAX)

    def _fetch_snapshot(self) -> Optional[dict]:
        """Return herdr's full live session snapshot in one socket call.

        `herdr api snapshot` returns result.snapshot with panes[]/tabs[]/
        workspaces[]. Each pane carries pane_id, terminal_id, agent_status,
        tab_id, workspace_id; each tab carries tab_id, label, workspace_id;
        each workspace carries workspace_id, label. Replaces the former
        pane-list + workspace-list + tab-list subprocess fan-out.

        Returns None on any failure (non-zero exit, timeout, missing binary,
        or malformed output). This is the single entry point all snapshot reads
        route through, so it swallows the same broad error set as the file's
        other herdr-subprocess helpers rather than letting one bad response
        kill the socket loop.
        """
        try:
            result = subprocess.run(
                ["herdr", "--session", self._herdr_session, "api", "snapshot"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # repr() the stderr: it can echo user-controlled labels/args and
                # may contain newlines/control chars that would otherwise forge
                # log lines. Matches the escaping used elsewhere in the codebase.
                logger.warning("Snapshot: `api snapshot` failed: %r", result.stderr)
                return None
            snapshot = json.loads(result.stdout)["result"]["snapshot"]
            if not isinstance(snapshot, dict):
                logger.warning("Snapshot: result.snapshot is not a dict; ignoring")
                return None
            if not self._snapshot_has_complete_identity(snapshot):
                logger.warning("Snapshot: identity collections are incomplete; ignoring")
                return None
            return snapshot
        except (
            subprocess.SubprocessError,
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as e:
            logger.warning(f"Snapshot: failed to fetch/parse: {e}")
            return None

    @staticmethod
    def _snapshot_has_complete_identity(snapshot: dict) -> bool:
        """Require complete identity collections before treating absence as proof."""
        required_fields = {
            "panes": ("pane_id",),
            "tabs": ("workspace_id", "label"),
            "workspaces": ("workspace_id", "label"),
        }
        for collection, fields in required_fields.items():
            records = snapshot.get(collection)
            if not isinstance(records, list):
                return False
            if any(
                not isinstance(record, dict) or any(not record.get(field) for field in fields)
                for record in records
            ):
                return False
        return True

    async def _reconcile(self) -> None:
        """Reconcile _pane_to_terminal map against live herdr state.

        Prunes stale pane entries, deletes orphaned DB terminal records,
        and kills workspaces with zero live terminals.
        """
        from cli_agent_orchestrator.backends.registry import get_backend
        from cli_agent_orchestrator.clients.database import (
            get_terminal_metadata,
            list_terminals_by_session,
        )
        from cli_agent_orchestrator.services.terminal_service import delete_missing_terminal

        # One socket call replaces the former pane-list + workspace-list +
        # tab-list subprocess fan-out. All three data structures below are derived
        # from this single snapshot.
        snapshot = self._fetch_snapshot()
        if snapshot is None or not self._snapshot_has_complete_identity(snapshot):
            logger.warning("Reconcile: no snapshot, skipping")
            return

        # Live pane_ids (from snapshot.panes).
        live_pane_ids = {p["pane_id"] for p in snapshot.get("panes", []) if p.get("pane_id")}

        # workspace_id -> label (= CAO session name), from snapshot.workspaces.
        # Skip malformed records (missing id/label) rather than letting a
        # KeyError escape _reconcile and kill the socket loop — matches the
        # defensive .get() style used in the tabs loop below.
        self._workspace_to_session = {
            ws["workspace_id"]: ws["label"]
            for ws in snapshot.get("workspaces", [])
            if ws.get("workspace_id") and ws.get("label")
        }

        # workspace_id -> set of live tab labels (= CAO window names), from
        # snapshot.tabs.
        live_tabs_by_workspace: Dict[str, set] = {}
        for tab in snapshot.get("tabs", []):
            ws_id = tab.get("workspace_id", "")
            label = tab.get("label", "")
            if ws_id and label:
                live_tabs_by_workspace.setdefault(ws_id, set()).add(label)

        # DB cross-check: find terminals in DB whose tab no longer exists in herdr.
        # This catches ghost records from previous server runs where _pane_to_terminal
        # starts empty (so the stale-pane diff below produces nothing).
        for ws_id, session_name in self._workspace_to_session.items():
            live_labels = live_tabs_by_workspace.get(ws_id, set())
            db_terminals = list_terminals_by_session(session_name)
            for term in db_terminals:
                window = term.get("tmux_window", "")
                if window and window not in live_labels:
                    logger.info(
                        f"Reconcile: deleting ghost terminal {term['id']} "
                        f"({session_name}:{window}) — tab not in herdr"
                    )
                    try:
                        delete_missing_terminal(
                            term["id"],
                            "Herdr snapshot reconciliation proved the persisted tab is missing",
                        )
                    except Exception as e:
                        logger.warning(
                            f"Reconcile: failed to delete ghost terminal {term['id']}: {e}"
                        )

        # Find stale panes: stored pane_id no longer in herdr's live pane list.
        #
        # A stale pane_id does NOT mean the terminal is dead. herdr renumbers
        # compact pane_ids when a sibling tab in the workspace closes, so a
        # still-running terminal's stored pane_id can fall out of the live list
        # while its tab is very much alive. Identity must come from the durable
        # tab label (tmux_window), never the ephemeral pane_id.
        stale_pane_ids = set(self._pane_to_terminal.keys()) - live_pane_ids
        if not stale_pane_ids:
            logger.debug("Reconcile: all panes live, nothing to prune")
            return

        # Live workspace labels, used to gate workspace teardown below: never
        # kill a workspace whose label is still present in herdr.
        live_workspace_labels = set(self._workspace_to_session.values())

        # Sessions that genuinely lost a terminal (deleted, not re-mapped).
        affected_sessions: Set[str] = set()
        remapped = 0
        deleted = 0

        for pane_id in stale_pane_ids:
            terminal_id = self._pane_to_terminal.get(pane_id)
            if not terminal_id:
                self._pane_to_terminal.pop(pane_id, None)
                continue

            # Session/window identity before any mutation.
            meta = get_terminal_metadata(terminal_id)
            term_session: Optional[str] = meta["tmux_session"] if meta else None
            term_window: Optional[str] = meta["tmux_window"] if meta else None

            # Re-map renumbered-but-live panes instead of deleting. A live tab
            # label means the pane_id was renumbered, not closed: re-resolve the
            # current pane_id and update both maps. Only when re-resolution fails
            # do we fall through to the delete path.
            label_live = self._label_still_live(term_window) if term_window else None
            if label_live is True:
                try:
                    # Invalidate pane cache so get_pane_id does a fresh label-based
                    # lookup instead of returning the stale pane_id we just proved
                    # is no longer live. See PR #309 review comment.
                    backend = get_backend()
                    if hasattr(backend, "_pane_cache"):
                        backend._pane_cache.pop(terminal_id, None)
                    new_pane_id = backend.get_pane_id(
                        terminal_id, term_session or "", term_window or ""
                    )
                except Exception as e:
                    logger.warning(
                        "Reconcile: tab %s live but pane re-resolve failed for %s (%s); "
                        "retaining until backend identity can be proven",
                        term_window,
                        terminal_id,
                        e,
                    )
                    continue
                else:
                    self._pane_to_terminal.pop(pane_id, None)
                    self._pane_to_terminal[new_pane_id] = terminal_id
                    self._terminal_to_pane[terminal_id] = new_pane_id
                    logger.info(
                        "Reconcile: re-mapped %s %s -> %s (pane renumbered, tab still live)",
                        terminal_id,
                        pane_id,
                        new_pane_id,
                    )
                    remapped += 1
                    continue

            if term_window and label_live is None:
                logger.warning(
                    "Reconcile: cannot prove whether tab %s for %s is missing; "
                    "retaining its report-recovery handle",
                    term_window,
                    terminal_id,
                )
                continue

            # The durable label is positively absent. Classify an assigned
            # worker as failed before deleting the irrecoverably missing pane.
            try:
                if delete_missing_terminal(
                    terminal_id,
                    "Herdr reconciliation proved the persisted worker tab is missing",
                ):
                    deleted += 1
                    self._pane_to_terminal.pop(pane_id, None)
                    self._terminal_to_pane.pop(terminal_id, None)
                    self._kiro_terminals.discard(terminal_id)
                    self._working_since.pop(terminal_id, None)
                    if term_session:
                        affected_sessions.add(term_session)
            except Exception as e:
                logger.warning(f"Reconcile: failed to delete terminal {terminal_id}: {e}")

        # Kill a workspace only when its label is gone from herdr AND no managed
        # terminal remains for the session. A live label means the workspace is
        # alive and its panes were merely renumbered — killing it would tear down
        # working agents.
        if affected_sessions:
            remaining_by_session: Dict[str, int] = {s: 0 for s in affected_sessions}
            for tid in self._terminal_to_pane:
                meta = get_terminal_metadata(tid)
                if meta and meta["tmux_session"] in remaining_by_session:
                    remaining_by_session[meta["tmux_session"]] += 1

            for session_name, remaining in remaining_by_session.items():
                if remaining == 0 and session_name not in live_workspace_labels:
                    try:
                        get_backend().kill_session(session_name)
                        logger.info(f"Reconcile: killed empty workspace {session_name}")
                    except Exception as e:
                        logger.warning(f"Reconcile: failed to kill workspace {session_name}: {e}")

        logger.info(
            "Reconcile: %d stale pane(s) — %d re-mapped, %d deleted",
            len(stale_pane_ids),
            remapped,
            deleted,
        )

    async def _connect(self) -> None:
        """Connect to the herdr socket."""
        self._reader, self._writer = await asyncio.open_unix_connection(self._socket_path)
        logger.info(f"Connected to herdr socket: {self._socket_path}")

    async def _subscribe_all_events(self) -> None:
        """Subscribe to all events in a SINGLE events.subscribe call.

        herdr (0.7.5) resets the entire connection when it receives a second
        events.subscribe on a connection that already has an active
        subscription, so this must remain exactly one events.subscribe per
        connection.

        The subscription is a broadcast pane.updated (sent with NO pane_id):
        herdr streams it for every pane and its payload carries agent_status,
        so a single broadcast subscription replaces the former per-pane
        pane.agent_status_changed subscriptions. This is independent of
        _pane_to_terminal — no per-pane enumeration is needed. The pane.closed
        and workspace.closed lifecycle events are sent in the same call.
        """
        subscriptions = [
            {"type": "pane.updated"},
            {"type": "pane.closed"},
            {"type": "workspace.closed"},
        ]
        message = {
            "id": "sub_all",
            "method": "events.subscribe",
            "params": {"subscriptions": subscriptions},
        }
        await self._send(message)
        logger.info(
            "Subscribed to broadcast pane.updated + lifecycle events "
            "in one events.subscribe call"
        )

    async def _event_loop(self) -> None:
        """Listen for events and dispatch delivery."""
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionError("Socket closed")

            try:
                event = json.loads(line.decode())
            except json.JSONDecodeError:
                continue

            # herdr identifies the event in the "event" key. Lifecycle events use
            # underscore names (pane_closed / workspace_closed); the agent-status
            # event uses the dotted name (pane.agent_status_changed). Normalize the
            # name so routing does not depend on the separator herdr happens to use.
            # (Older code read "type" and matched dotted lifecycle names, which never
            # matched herdr's real wire format — lifecycle cleanup silently never ran.)
            raw_event = event.get("event", "") or event.get("type", "")
            event_name = raw_event.replace("_", ".")

            # Handle lifecycle events
            if event_name in ("pane.closed", "workspace.closed"):
                self._handle_lifecycle_event(event_name, event.get("data", {}))
                continue

            data = event.get("data", {})
            # Broadcast pane.updated nests the pane under data.pane; retired
            # agent-status events used top-level data. Fall back to data, and
            # guard against a null/non-dict pane so one malformed event cannot
            # escape the loop and kill delivery.
            pane_obj = data.get("pane") or data
            if not isinstance(pane_obj, dict):
                pane_obj = {}
            pane_id = pane_obj.get("pane_id", "")
            status = pane_obj.get("agent_status", "")

            # Only process events for managed panes
            terminal_id = self._pane_to_terminal.get(pane_id)
            if not terminal_id:
                continue

            if status in ("idle", "done"):
                # Clear working timestamp
                self._working_since.pop(terminal_id, None)
                # Trigger delivery
                self._deliver(terminal_id)

            elif status == "working":
                # Track working start for kiro supplement check
                if terminal_id in self._kiro_terminals:
                    if terminal_id not in self._working_since:
                        self._working_since[terminal_id] = time.time()

    def _label_still_live(self, window_name: str) -> Optional[bool]:
        """Return tab liveness, or None when the backend cannot prove it.

        Used to disambiguate herdr's reused compact pane_ids on replayed
        pane_closed events. The tab label is unique per incarnation, so a live
        label means the close event refers to an older incarnation and is stale.

        Query failures fail closed toward retention.  Treating uncertainty as a
        missing backend could destroy the only uncaptured report handle.
        """
        try:
            result = subprocess.run(
                ["herdr", "--session", self._herdr_session, "tab", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "_label_still_live: herdr tab list failed (rc=%s): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
                return None
            tab_data = json.loads(result.stdout)
            result_data = tab_data.get("result")
            if not isinstance(result_data, dict):
                return None
            tabs = result_data.get("tabs")
            if not isinstance(tabs, list) or any(
                not isinstance(tab, dict) or not tab.get("label") for tab in tabs
            ):
                return None
            live_labels = {tab["label"] for tab in tabs}
            return window_name in live_labels
        except (
            subprocess.SubprocessError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            TypeError,
        ) as e:
            logger.warning("_label_still_live: could not query herdr (%s)", e)
            return None

    def _workspace_label_still_live(self, session_name: str) -> Optional[bool]:
        """Return workspace-label liveness, or None when absence is unproven."""
        snapshot = self._fetch_snapshot()
        if snapshot is None:
            return None
        return any(workspace.get("label") == session_name for workspace in snapshot["workspaces"])

    def _handle_lifecycle_event(self, event_type: str, data: dict) -> None:
        """Handle pane.closed and workspace.closed events."""
        from cli_agent_orchestrator.clients.database import (
            get_terminal_metadata,
            list_terminals_by_session,
        )
        from cli_agent_orchestrator.services.terminal_service import delete_missing_terminal

        if event_type == "pane.closed":
            pane_id = data.get("pane_id", "")
            terminal_id = self._pane_to_terminal.get(pane_id)
            if not terminal_id:
                return

            # Get session before cleanup
            meta = get_terminal_metadata(terminal_id)
            session_name = meta["tmux_session"] if meta else None

            # Guard against herdr's compact pane_id reuse + event replay.
            #
            # herdr (0.6.8) reuses compact pane_ids when a tab is killed and a
            # new tab takes the same index, AND replays the ENTIRE pane_closed
            # history on every fresh events.subscribe (e.g. after a reconnect on
            # socket disconnect). So a replayed close for an OLD incarnation of
            # this pane_id arrives mapped to the terminal that now occupies the
            # reused index — deleting a live terminal.
            #
            # The tab label (tmux_window) is unique per incarnation, so confirm
            # the label is genuinely gone from herdr before deleting. If the
            # label is still live, this close is stale (replayed) — ignore it.
            # If herdr cannot be queried, retain the recovery handle until the
            # backend can positively prove that this incarnation is gone.
            window_name = meta["tmux_window"] if meta else None
            label_live = self._label_still_live(window_name) if window_name else None
            if label_live is True:
                logger.info(
                    "pane.closed: ignoring stale close for %s (pane=%s) — "
                    "label %s still live in herdr (compact pane_id reused)",
                    terminal_id,
                    pane_id,
                    window_name,
                )
                return
            if window_name and label_live is None:
                logger.warning(
                    "pane.closed: cannot verify label %s for %s; retaining its "
                    "report-recovery handle",
                    window_name,
                    terminal_id,
                )
                return

            # A pane.closed event plus a positively absent durable label proves
            # the backend handle is gone. The missing-terminal path records a
            # failed outcome before deleting an uncaptured assigned worker.
            deleted = False
            try:
                deleted = delete_missing_terminal(
                    terminal_id,
                    "Herdr pane.closed event and absent durable tab label proved the pane missing",
                )
                if not deleted:
                    logger.warning("pane.closed: cleanup deferred for terminal %s", terminal_id)
            except Exception as e:
                logger.warning(f"pane.closed: failed to delete terminal {terminal_id}: {e}")

            if not deleted:
                return

            self._pane_to_terminal.pop(pane_id, None)
            self._terminal_to_pane.pop(terminal_id, None)
            self._kiro_terminals.discard(terminal_id)
            self._working_since.pop(terminal_id, None)

            logger.info(f"pane.closed: cleaned up terminal {terminal_id} (pane={pane_id})")

            # A pane-close event proves only that pane is gone. Workspace
            # teardown belongs to a verified workspace-close/reconcile path; it
            # must not kill other unregistered or not-yet-mapped panes.

        elif event_type == "workspace.closed":
            workspace_id = data.get("workspace_id", "")
            session_name = self._workspace_to_session.get(workspace_id)
            if not session_name:
                # Once the workspace is absent, a fresh backend query cannot map
                # its opaque id back to a persisted session safely. Guessing from
                # a live/reused id could delete a new workspace, so retain rows.
                logger.warning(
                    "workspace.closed: no persisted session mapping for %s; retaining rows",
                    workspace_id,
                )
                return

            workspace_live = self._workspace_label_still_live(session_name)
            if workspace_live is True:
                logger.info(
                    "workspace.closed: ignoring stale close for live session %s",
                    session_name,
                )
                return
            if workspace_live is None:
                logger.warning(
                    "workspace.closed: cannot prove session %s is absent; retaining rows",
                    session_name,
                )
                return

            # The event plus a positively absent durable workspace label proves
            # every contained backend pane is gone. Classify workers first.
            deleted_terminal_ids = set()
            cleanup_complete = True
            for terminal in list_terminals_by_session(session_name):
                terminal_id = terminal["id"]
                try:
                    if delete_missing_terminal(
                        terminal_id,
                        "Herdr workspace.closed event proved the backend workspace missing",
                    ):
                        deleted_terminal_ids.add(terminal_id)
                    else:
                        cleanup_complete = False
                        logger.warning(
                            "workspace.closed: cleanup deferred for terminal %s", terminal_id
                        )
                except Exception as e:
                    cleanup_complete = False
                    logger.warning(
                        "workspace.closed: failed to cleanup terminal %s: %s", terminal_id, e
                    )

            # Prune maps for terminals belonging to this session. Match on each
            # terminal's DB session rather than a pane_id/workspace_id string
            # prefix: herdr renumbers compact pane_ids and does not guarantee
            # they begin with the workspace_id, so a prefix test is unreliable.
            # This mirrors the session match used in the pane.closed handler.
            to_remove = [
                (pid, tid)
                for pid, tid in self._pane_to_terminal.items()
                if tid in deleted_terminal_ids
            ]
            for pid, tid in to_remove:
                self._pane_to_terminal.pop(pid, None)
                self._terminal_to_pane.pop(tid, None)
                self._kiro_terminals.discard(tid)
                self._working_since.pop(tid, None)

            if cleanup_complete:
                self._workspace_to_session.pop(workspace_id, None)
            logger.info(
                "workspace.closed: processed session %s (%d terminals deleted; complete=%s)",
                session_name,
                len(to_remove),
                cleanup_complete,
            )

    # TODO: _deliver() calls callback synchronously — if callback is async,
    # this will need a threadsafe bridge (out of scope for this change).
    def _deliver(self, terminal_id: str) -> None:
        """Check and deliver pending messages for a terminal."""
        if self._delivery_callback:
            try:
                self._delivery_callback(terminal_id)
            except Exception as e:
                logger.error(f"Delivery failed for terminal {terminal_id}: {e}")

    async def check_kiro_supplements(self) -> None:
        """Periodic check for kiro-cli terminals stuck in 'working' state.

        For terminals in 'working' for >30s, read pane content and check
        for permission prompt patterns.
        """
        import subprocess

        now = time.time()
        for terminal_id in list(self._working_since.keys()):
            if terminal_id not in self._kiro_terminals:
                continue

            working_duration = now - self._working_since[terminal_id]
            if working_duration < _KIRO_WORKING_THRESHOLD:
                continue

            # Read pane and check for permission prompt
            pane_id = self._terminal_to_pane.get(terminal_id)
            if not pane_id:
                continue

            result = subprocess.run(
                ["herdr", "--session", self._herdr_session, "pane", "read", pane_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                continue

            # Check for kiro permission prompt pattern
            # (WAITING_USER_ANSWER indicator)
            from cli_agent_orchestrator.providers.kiro_cli import TUI_PERMISSION_PATTERN

            if re.search(TUI_PERMISSION_PATTERN, result.stdout):
                logger.info(
                    f"Kiro permission prompt detected for {terminal_id} "
                    f"(working for {working_duration:.0f}s)"
                )
                self._deliver(terminal_id)
                # Reset the timer so we don't spam
                self._working_since[terminal_id] = now

    async def _send(self, message: dict) -> None:
        """Send a JSON message to the herdr socket."""
        assert self._writer is not None
        data = json.dumps(message).encode() + b"\n"
        self._writer.write(data)
        await self._writer.drain()
