"""Opt-in, isolated E2E for real Claude Code assigned-worker callbacks.

The test redirects every CAO-owned path to ``tmp_path`` while deliberately
using the operator-selected Claude HOME for provider authentication. It copies
only the named agent profiles into the isolated CAO store, launches a uniquely
named tmux session, and never starts/stops the production CAO server or opens
the production CAO database.

Run explicitly with::

    CAO_RUN_ACTUAL_CLAUDE_E2E=1 \
    CAO_ACTUAL_CLAUDE_HOME=/path/to/provider-home \
    CAO_ACTUAL_AGENT_STORE=/path/to/cao/agent-store \
    pytest -m e2e test/services/test_claude_code_completion_callback_actual_e2e.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from test.fixtures.cao_server import _pick_free_port, _start_cao_server
from unittest.mock import patch

import pytest
import requests

from cli_agent_orchestrator.models.provider_completion import ProviderCompletionReport
from cli_agent_orchestrator.services import provider_completion_report as completion_reports

CLAUDE_PROFILES = (
    "atlas_data_worker",
    "atlas_equities_worker",
    "atlas_claude_worker",
    "atlas_ops_support",
)
SYNTHETIC_TASK = (
    "SYNTHETIC CLAUDE CALLBACK SMOKE TEST ONLY.\n"
    "Do not call send_message.\n"
    "Return exactly:\n"
    "SYNTHETIC_CLAUDE_CALLBACK_SMOKE_OK"
)
SYNTHETIC_RESULT = "SYNTHETIC_CLAUDE_CALLBACK_SMOKE_OK"
TASK_SHA256 = hashlib.sha256(SYNTHETIC_TASK.encode("utf-8")).hexdigest()
RESULT_SHA256 = hashlib.sha256(SYNTHETIC_RESULT.encode("utf-8")).hexdigest()


def _wait_for_callback(database_path: Path, worker_id: str, timeout: float = 180.0) -> tuple:
    """Observe durable server state; the supervisor itself never polls."""
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT c.completion_id, c.delivery_state, c.attempt_count, "
                    "c.final_result, c.final_result_sha256, c.inbox_message_id, "
                    "i.status, i.origin "
                    "FROM assigned_worker_callbacks AS c "
                    "LEFT JOIN inbox AS i ON i.id = c.inbox_message_id "
                    "WHERE c.worker_terminal_id = ?",
                    (worker_id,),
                ).fetchone()
                callback_count = connection.execute(
                    "SELECT COUNT(*) FROM assigned_worker_callbacks "
                    "WHERE worker_terminal_id = ?",
                    (worker_id,),
                ).fetchone()[0]
                inbox_count = connection.execute(
                    "SELECT COUNT(*) FROM assigned_worker_callbacks AS c "
                    "JOIN inbox AS i ON i.id = c.inbox_message_id "
                    "WHERE c.worker_terminal_id = ? AND i.origin = 'server_completion'",
                    (worker_id,),
                ).fetchone()[0]
            last_state = (row, callback_count, inbox_count)
            if (
                row is not None
                and row[1] == "acknowledged"
                and row[6] == "delivered"
                and callback_count == 1
                and inbox_count == 1
            ):
                return row
        except sqlite3.OperationalError:
            pass
        time.sleep(0.1)
    raise AssertionError(
        f"Claude callback was not durably delivered in time; last state={last_state!r}"
    )


def _load_report(cao_home: Path, worker_id: str, completion_id: str) -> ProviderCompletionReport:
    report_root = cao_home / "provider-completion-reports"
    with patch.object(completion_reports, "PROVIDER_COMPLETION_REPORT_DIR", report_root):
        return completion_reports.load_completion_report("claude_code", worker_id, completion_id)


def _counts(database_path: Path, worker_id: str, caller_id: str) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as connection:
        callback_rows = connection.execute(
            "SELECT COUNT(*) FROM assigned_worker_callbacks WHERE worker_terminal_id = ?",
            (worker_id,),
        ).fetchone()[0]
        callback_inbox_rows = connection.execute(
            "SELECT COUNT(*) FROM assigned_worker_callbacks AS c "
            "JOIN inbox AS i ON i.id = c.inbox_message_id "
            "WHERE c.worker_terminal_id = ? AND i.origin = 'server_completion'",
            (worker_id,),
        ).fetchone()[0]
        explicit_worker_rows = connection.execute(
            "SELECT COUNT(*) FROM inbox "
            "WHERE sender_id = ? AND receiver_id = ? AND origin = 'explicit'",
            (worker_id, caller_id),
        ).fetchone()[0]
    return int(callback_rows), int(callback_inbox_rows), int(explicit_worker_rows)


@pytest.mark.e2e
def test_real_claude_profiles_complete_without_send_message_and_recover_after_restart(
    tmp_path: Path,
) -> None:
    if os.environ.get("CAO_RUN_ACTUAL_CLAUDE_E2E") != "1":
        pytest.skip("set CAO_RUN_ACTUAL_CLAUDE_E2E=1 for the real Claude provider E2E")
    if shutil.which("tmux") is None or shutil.which("claude") is None:
        pytest.skip("tmux and claude are required for the real Claude provider E2E")

    real_home_raw = os.environ.get("CAO_ACTUAL_CLAUDE_HOME")
    profile_store_raw = os.environ.get("CAO_ACTUAL_AGENT_STORE")
    if not real_home_raw or not profile_store_raw:
        pytest.skip("CAO_ACTUAL_CLAUDE_HOME and CAO_ACTUAL_AGENT_STORE are required")
    real_home = Path(real_home_raw).resolve()
    profile_store = Path(profile_store_raw).resolve()
    if not real_home.is_dir() or not profile_store.is_dir():
        pytest.skip("the selected real Claude home/profile store is unavailable")

    version = subprocess.run(
        ["claude", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    assert version == "2.1.259 (Claude Code)"

    isolated_root = tmp_path / "actual-claude-isolated"
    # Match the fixture's returned db_path while still pinning the same path
    # explicitly through CAO_HOME_DIR when HOME is switched for Claude auth.
    cao_home = isolated_root / ".aws" / "cli-agent-orchestrator"
    isolated_profile_store = cao_home / "agent-store"
    isolated_profile_store.mkdir(parents=True)
    for profile_name in CLAUDE_PROFILES:
        source = profile_store / f"{profile_name}.md"
        assert source.is_file(), f"missing configured profile: {profile_name}"
        shutil.copy2(source, isolated_profile_store / source.name)

    working_directory = isolated_root / "worker-cwd"
    working_directory.mkdir(parents=True)
    port = _pick_free_port()
    server = _start_cao_server(
        isolated_root,
        port,
        extra_env={
            # Claude reads its existing provider authentication from this HOME.
            # Every CAO-owned path remains rooted at the isolated override.
            "HOME": str(real_home),
            "CAO_HOME_DIR": str(cao_home),
        },
        deadline=20.0,
    )
    restarted_server = None
    session_name = f"claude-completion-{uuid.uuid4().hex[:8]}"
    actual_session = None
    evidence: dict[str, tuple[str, ProviderCompletionReport]] = {}
    try:
        branch_src = Path(__file__).resolve().parents[2] / "src"
        supervisor_response = requests.post(
            f"{server.url}/sessions",
            params={
                "provider": "mock_cli",
                "agent_profile": "developer",
                "session_name": session_name,
                "working_directory": str(working_directory),
            },
            # The test runner may intentionally use a venv installed from a
            # different checkout. Pin the unique tmux session to this branch's
            # source so the launcher's ``python -m`` resolves the code under test.
            json={"env_vars": {"PYTHONPATH": str(branch_src)}},
            timeout=20,
        )
        assert supervisor_response.status_code in (200, 201), supervisor_response.text
        supervisor = supervisor_response.json()
        supervisor_id = supervisor["id"]
        actual_session = supervisor["session_name"]

        for profile_name in CLAUDE_PROFILES:
            worker_response = requests.post(
                f"{server.url}/sessions/{actual_session}/terminals",
                params={
                    "provider": "claude_code",
                    "agent_profile": profile_name,
                    "caller_id": supervisor_id,
                    "defer_init": "true",
                    "working_directory": str(working_directory),
                },
                json={
                    "initial_message": SYNTHETIC_TASK,
                    "initial_message_orchestration_type": "assign",
                },
                timeout=15,
            )
            assert worker_response.status_code == 201, worker_response.text
            worker_id = worker_response.json()["id"]

            callback = _wait_for_callback(server.db_path, worker_id)
            completion_id = callback[0]
            assert callback[1:] == (
                "acknowledged",
                1,
                SYNTHETIC_RESULT,
                RESULT_SHA256,
                callback[5],
                "delivered",
                "server_completion",
            )
            assert callback[5] is not None
            assert _counts(server.db_path, worker_id, supervisor_id) == (1, 1, 0)

            report = _load_report(cao_home, worker_id, completion_id)
            assert report.provider == "claude_code"
            assert report.terminal_id == worker_id
            assert report.completion_id == completion_id
            assert report.completion_state == "success"
            assert report.provider_result_subtype == "success"
            assert report.provider_terminal_reason == "completed"
            assert report.provider_is_error is False
            assert report.dispatched_input_sha256 == TASK_SHA256
            assert report.final_response == SYNTHETIC_RESULT
            assert report.final_response_sha256 == RESULT_SHA256
            assert report.provider_input_id is not None
            assert report.provider_turn_id
            assert report.provider_session_id
            evidence[worker_id] = (completion_id, report)

        delete_response = requests.delete(f"{server.url}/sessions/{actual_session}", timeout=30)
        assert delete_response.status_code == 200, delete_response.text
        actual_session = None
        server.stop()

        restarted_server = _start_cao_server(
            isolated_root,
            port,
            extra_env={"HOME": str(real_home), "CAO_HOME_DIR": str(cao_home)},
            deadline=20.0,
        )
        time.sleep(1.0)
        for worker_id, (completion_id, original_report) in evidence.items():
            assert _counts(restarted_server.db_path, worker_id, supervisor_id) == (1, 1, 0)
            assert _load_report(cao_home, worker_id, completion_id) == original_report
    finally:
        if actual_session is not None:
            try:
                requests.delete(f"{server.url}/sessions/{actual_session}", timeout=20)
            except requests.RequestException:
                pass
        server.stop()
        if restarted_server is not None:
            restarted_server.stop()
