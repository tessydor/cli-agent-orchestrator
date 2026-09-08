"""Snapshot-bound native Claude menu navigation. Never paste answers into menus.

This transport does not confer approval authority. Supervisors may answer only
questions delegated to them; owner identity/permission decisions remain human.
"""

import hashlib
import re
import threading
import time
from typing import Any

_LOCK = threading.RLock()
_PENDING: dict[tuple[str, str], tuple[str, float]] = {}
_CONSUMED: dict[tuple[str, str], str] = {}
_ROW = re.compile(r"^\s*([❯>›]?)\s*(\d+)\. (.+)$")


def parse_screen(screen: str) -> tuple[str, list[tuple[int, str]], int]:
    lines = screen.splitlines()
    if not any("↑/↓ to navigate" in line or "↑/↓ to select" in line for line in lines[-5:]):
        raise ValueError("No current Claude selection menu")
    rows = []
    normalized = []
    selected = []
    for line in lines:
        match = _ROW.match(line)
        if match:
            marker, number, label = match.groups()
            number = int(number)
            rows.append((number, label))
            if marker:
                selected.append(number)
            normalized.append(f"{number}. {label}")
        else:
            normalized.append(line.rstrip())
    if (
        not rows
        or len(rows) > 9
        or [n for n, _ in rows] != list(range(1, len(rows) + 1))
        or len(selected) != 1
    ):
        raise ValueError("Ambiguous or incomplete Claude menu")
    digest = hashlib.sha256("\n".join(normalized).encode()).hexdigest()
    return digest, rows, selected[0]


def _screen(terminal_id: str, caller_id: str) -> str:
    from cli_agent_orchestrator.services import terminal_service as terminals

    worker = terminals.get_terminal_metadata(terminal_id)
    caller = terminals.get_terminal_metadata(caller_id)
    if (
        not worker
        or not caller
        or worker.get("caller_id") != caller_id
        or worker["tmux_session"] != caller["tmux_session"]
        or worker["provider"] != "claude_code"
    ):
        raise ValueError("Question requires a direct Claude worker in this session")
    backend = terminals.get_backend()
    if backend.get_pane_current_command(worker["tmux_session"], worker["tmux_window"]) != "claude":
        raise ValueError("Worker is not in native Claude")
    return backend.get_history(
        worker["tmux_session"], worker["tmux_window"], visible_only=True, strip_escapes=True
    )


def snapshot(terminal_id: str, caller_id: str) -> dict[str, Any]:
    with _LOCK:
        digest, rows, selected = parse_screen(_screen(terminal_id, caller_id))
        if _CONSUMED.get((terminal_id, caller_id)) == digest:
            raise ValueError("This menu was already answered; wait for a changed question")
        _PENDING[(terminal_id, caller_id)] = (digest, time.monotonic() + 120)
        return {
            "prompt_sha256": digest,
            "options": [{"index": n, "label": label} for n, label in rows],
            "selected_index": selected,
            "expires_in_seconds": 120,
            "authority": "Only delegated routine questions; never human identity or owner approval.",
        }


def answer(terminal_id: str, caller_id: str, prompt_sha256: str, index: int) -> dict[str, Any]:
    from cli_agent_orchestrator.services import terminal_service as terminals

    if not isinstance(caller_id, str) or not isinstance(prompt_sha256, str):
        raise ValueError("Invalid question identity")
    with _LOCK:
        pending = _PENDING.get((terminal_id, caller_id))
        if not pending or pending[0] != prompt_sha256 or pending[1] < time.monotonic():
            raise ValueError("Question snapshot missing, stale, or already consumed")
        digest, rows, selected = parse_screen(_screen(terminal_id, caller_id))
        if digest != prompt_sha256 or type(index) is not int or not 1 <= index <= len(rows):
            raise ValueError("Menu changed or option is invalid")
        label = dict(rows)[index]
        if label.lower().startswith(("type something", "chat about", "other")):
            raise ValueError("Free-form menu entries require direct interaction")
        # Consume before the first key: errors/timeouts must never replay Enter.
        del _PENDING[(terminal_id, caller_id)]
        _CONSUMED[(terminal_id, caller_id)] = digest
        while selected != index:
            expected = selected + (1 if index > selected else -1)
            terminals.send_special_key(terminal_id, "Down" if index > selected else "Up")
            deadline = time.monotonic() + 1
            while True:
                digest, _, observed = parse_screen(_screen(terminal_id, caller_id))
                if digest != prompt_sha256:
                    raise ValueError("Menu changed during navigation; Enter was not sent")
                if observed == expected:
                    selected = observed
                    break
                if observed != selected or time.monotonic() >= deadline:
                    raise ValueError("Menu navigation not confirmed; Enter was not sent")
                time.sleep(0.025)
        terminals.send_special_key(terminal_id, "Enter")
        return {"success": True, "selected_index": index, "label": label}
