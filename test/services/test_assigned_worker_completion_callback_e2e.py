"""Isolated process-level proof for server-generated assigned callbacks.

This test uses the deterministic mock provider, a temporary HOME/database, a
fresh localhost port, and managed subprocesses.  It never contacts or restarts an
operator's production cao-server.
"""

import hashlib
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from test.fixtures.cao_server import _pick_free_port, _start_cao_server

import pytest
import requests


def _wait_for_one_delivered_callback(
    database_path: Path, caller_id: str, timeout: float = 30.0
) -> tuple:
    """Observe server-owned durable state without polling the supervisor API."""
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT c.delivery_state, c.attempt_count, c.final_result, "
                    "c.final_result_sha256, "
                    "i.id, i.status, i.origin "
                    "FROM assigned_worker_callbacks AS c "
                    "JOIN inbox AS i ON i.id = c.inbox_message_id "
                    "WHERE c.caller_id = ?",
                    (caller_id,),
                ).fetchone()
                count = connection.execute(
                    "SELECT COUNT(*) FROM inbox "
                    "WHERE receiver_id = ? AND origin = 'server_completion'",
                    (caller_id,),
                ).fetchone()[0]
            last_state = (row, count)
            if row is not None and row[0] == "acknowledged" and row[5] == "delivered":
                assert count == 1
                return row
        except sqlite3.OperationalError:
            # Schema creation and the first callback transaction can briefly
            # overlap this external observer.
            pass
        time.sleep(0.05)
    raise AssertionError(f"callback was not durably delivered in time; last state={last_state!r}")


def _server_callback_count(database_path: Path, caller_id: str) -> int:
    with sqlite3.connect(database_path) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM inbox "
                "WHERE receiver_id = ? AND origin = 'server_completion'",
                (caller_id,),
            ).fetchone()[0]
        )


def _synthetic_callback_row_counts(
    database_path: Path, worker_id: str, caller_id: str
) -> tuple[int, int, int]:
    """Return callback, linked-inbox, and explicit-worker-message counts."""
    with sqlite3.connect(database_path) as connection:
        callback_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM assigned_worker_callbacks WHERE worker_terminal_id = ?",
                (worker_id,),
            ).fetchone()[0]
        )
        linked_inbox_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM assigned_worker_callbacks AS c "
                "JOIN inbox AS i ON i.id = c.inbox_message_id "
                "WHERE c.worker_terminal_id = ?",
                (worker_id,),
            ).fetchone()[0]
        )
        explicit_worker_messages = int(
            connection.execute(
                "SELECT COUNT(*) FROM inbox "
                "WHERE sender_id = ? AND receiver_id = ? AND origin = 'explicit'",
                (worker_id, caller_id),
            ).fetchone()[0]
        )
    return callback_rows, linked_inbox_rows, explicit_worker_messages


@pytest.mark.e2e
def test_no_send_completion_arrives_once_and_restart_does_not_duplicate(tmp_path):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is required for the isolated callback E2E")
    if shutil.which("mock_cli") is None:
        pytest.skip("the repository mock_cli fixture is not on PATH")

    isolated_home = tmp_path / "isolated-cao-home"
    port = _pick_free_port()
    first_server = _start_cao_server(isolated_home, port, deadline=15.0)
    restarted_server = None
    session_name = f"callback-final-result-{uuid.uuid4().hex[:8]}"
    actual_session = None
    try:
        supervisor_response = requests.post(
            f"{first_server.url}/sessions",
            params={
                "provider": "mock_cli",
                "agent_profile": "developer",
                "session_name": session_name,
            },
            timeout=20,
        )
        assert supervisor_response.status_code in (200, 201), supervisor_response.text
        supervisor = supervisor_response.json()
        supervisor_id = supervisor["id"]
        actual_session = supervisor["session_name"]

        worker_response = requests.post(
            f"{first_server.url}/sessions/{actual_session}/terminals",
            params={
                "provider": "mock_cli",
                "agent_profile": "developer",
                "caller_id": supervisor_id,
                "defer_init": "true",
            },
            json={
                "initial_message": "Reply with exactly SYNTHETIC_CALLBACK_SMOKE_OK",
                "initial_message_orchestration_type": "assign",
            },
            timeout=10,
        )
        assert worker_response.status_code == 201, worker_response.text
        worker_id = worker_response.json()["id"]

        callback = _wait_for_one_delivered_callback(first_server.db_path, supervisor_id)
        assert callback[1] == 1
        assert callback[2] == "SYNTHETIC_CALLBACK_SMOKE_OK"
        assert callback[3] == hashlib.sha256(b"SYNTHETIC_CALLBACK_SMOKE_OK").hexdigest()
        assert callback[6] == "server_completion"
        # The one callback links to exactly one server-generated inbox row,
        # and there is no explicit worker->supervisor message from which the
        # result could have been copied.
        assert _synthetic_callback_row_counts(first_server.db_path, worker_id, supervisor_id) == (
            1,
            1,
            0,
        )

        # The mock's actual assistant output differs from the assignment prompt.
        # This independent display read is evidence only; callback capture above
        # came from the mock's structured provider report.
        worker_output = requests.get(
            f"{first_server.url}/terminals/{worker_id}/output",
            params={"mode": "last"},
            timeout=5,
        )
        assert worker_output.status_code == 200, worker_output.text
        assert worker_output.json()["output"] == "SYNTHETIC_CALLBACK_SMOKE_OK"

        # This is the first and only supervisor read. Arrival was driven by the
        # server's status event and inbox wakeup, not supervisor polling.
        time.sleep(0.5)
        output_response = requests.get(
            f"{first_server.url}/terminals/{supervisor_id}/output",
            params={"mode": "full"},
            timeout=5,
        )
        assert output_response.status_code == 200, output_response.text
        assert "SYNTHETIC_CALLBACK_SMOKE_OK" in output_response.json()["output"]
        assert "Reply with exactly" not in output_response.json()["output"]

        audit_response = requests.get(
            f"{first_server.url}/assigned-workers/{worker_id}/completion-callback",
            timeout=5,
        )
        assert audit_response.status_code == 200, audit_response.text
        assert audit_response.json()["delivery_state"] == "acknowledged"

        # Tear down only the isolated synthetic session before stopping its
        # isolated server; callback/report rows intentionally survive.
        delete_response = requests.delete(
            f"{first_server.url}/sessions/{actual_session}", timeout=10
        )
        assert delete_response.status_code == 200, delete_response.text
        actual_session = None
        first_server.stop()

        restarted_server = _start_cao_server(isolated_home, port, deadline=15.0)
        time.sleep(1.0)
        assert _server_callback_count(restarted_server.db_path, supervisor_id) == 1
        assert _synthetic_callback_row_counts(
            restarted_server.db_path, worker_id, supervisor_id
        ) == (1, 1, 0)
        with sqlite3.connect(restarted_server.db_path) as connection:
            state = connection.execute(
                "SELECT delivery_state, attempt_count, final_result, final_result_sha256 "
                "FROM assigned_worker_callbacks WHERE worker_terminal_id = ?",
                (worker_id,),
            ).fetchone()
        assert state == (
            "acknowledged",
            1,
            "SYNTHETIC_CALLBACK_SMOKE_OK",
            hashlib.sha256(b"SYNTHETIC_CALLBACK_SMOKE_OK").hexdigest(),
        )
    finally:
        if actual_session is not None:
            try:
                requests.delete(f"{first_server.url}/sessions/{actual_session}", timeout=5)
            except requests.RequestException:
                pass
        first_server.stop()
        if restarted_server is not None:
            restarted_server.stop()
