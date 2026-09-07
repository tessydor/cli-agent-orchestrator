"""Real tmux/PTY regressions for assigned Claude stream-json input.

These tests deliberately replace only the networked ``claude`` executable with
a local byte-capturing child. The production provider encoder, tmux paste path,
launcher process, terminal line discipline, and launcher-to-child pipe all
remain in the exercised path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.clients.tmux import TmuxClient
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
from cli_agent_orchestrator.services import claude_completion_launcher as launcher

TERMINAL_ID = "c1a0de01"
COMPLETION_ID = "1234567890abcdef1234567890abcdef"
RECORD_LENGTHS = (4093, 4095, 4096, 4097, 32768)
SOURCE_ROOT = Path(launcher.__file__).resolve().parents[2]
FOLLOWUP_INPUT_ID = uuid.UUID("99999999-8888-4777-8666-555555555555")


def _exact_encoded_record(record_bytes: int, orchestration_type: str) -> str:
    """Build an exact-size record containing every relevant escaping case."""
    provider = ClaudeCodeProvider(
        TERMINAL_ID,
        "cao-jsonl-pty-test",
        "worker",
        completion_id=COMPLETION_ID,
    )
    structured_prefix = 'line one\nline two: αβ雪; quote="yes"; slash=\\; end\n'
    with patch(
        "cli_agent_orchestrator.providers.claude_code.uuid.uuid4",
        return_value=FOLLOWUP_INPUT_ID,
    ):
        baseline = provider.encode_terminal_input(structured_prefix, orchestration_type)
        filler_bytes = record_bytes - len(baseline.encode("utf-8"))
        assert filler_bytes >= 0

        encoded = provider.encode_terminal_input(
            structured_prefix + ("x" * filler_bytes), orchestration_type
        )
    assert len(encoded.encode("utf-8")) == record_bytes
    return encoded


def _write_fake_claude(path: Path) -> None:
    """Create a no-network stream-json peer that atomically captures one record."""
    path.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import json\n"
        + "import os\n"
        + "import sys\n"
        + "request = json.loads(sys.stdin.buffer.readline())\n"
        + 'response = {"type": "control_response", "response": '
        + '{"subtype": "success", "request_id": request["request_id"], '
        + '"response": {}}}\n'
        + 'sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\\n")\n'
        + "sys.stdout.flush()\n"
        + "record = sys.stdin.buffer.readline()\n"
        + 'target = os.environ["CAO_TEST_CAPTURE"]\n'
        + 'temporary = target + ".tmp"\n'
        + 'with open(temporary, "wb") as capture:\n'
        + "    capture.write(record)\n"
        + "os.replace(temporary, target)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _wait_for_ready(target: str, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        capture = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", target, "-S", "-100"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if launcher.ADAPTER_READY_MARKER in capture.stdout:
            return
        time.sleep(0.05)
    raise AssertionError("Claude stream launcher did not publish its ready marker")


def _wait_for_capture(path: Path, timeout: float = 8.0) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            time.sleep(0.05)
    raise AssertionError("fake Claude child did not capture a stream-json record")


def _round_trip_through_tmux(tmp_path: Path, encoded: str) -> bytes:
    """Deliver one encoded record through the production tmux/launcher path."""
    capture_path = tmp_path / "child-record.jsonl"
    fake_claude = tmp_path / "claude"
    _write_fake_claude(fake_claude)

    session_name = f"cao-jsonl-{uuid.uuid4().hex[:12]}"
    window_name = "worker"
    target = f"{session_name}:{window_name}"
    command = shlex.join(
        [
            shutil.which("env") or "/usr/bin/env",
            f"PYTHONPATH={SOURCE_ROOT}",
            f"CAO_HOME_DIR={tmp_path / 'cao-home'}",
            f"CAO_TEST_CAPTURE={capture_path}",
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.claude_completion_launcher",
            "--terminal-id",
            TERMINAL_ID,
            "--completion-id",
            COMPLETION_ID,
            "--",
            str(fake_claude),
        ]
    )

    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session_name,
                "-n",
                window_name,
                command,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _wait_for_ready(target)

        # This is the production CAO delivery operation: load-buffer,
        # paste-buffer -p, then exactly one Enter for the JSONL boundary.
        TmuxClient().send_keys(
            session_name,
            window_name,
            encoded,
            enter_count=1,
            force_bracketed_paste=False,
            submit_delay=0.0,
        )
        return _wait_for_capture(capture_path)
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
@pytest.mark.parametrize(
    "orchestration_type",
    ("assign", "send_message"),
    ids=("initial-assignment", "follow-up"),
)
@pytest.mark.parametrize("record_bytes", RECORD_LENGTHS)
def test_stream_json_record_is_byte_exact_through_real_tmux_pty(
    tmp_path: Path,
    orchestration_type: str,
    record_bytes: int,
) -> None:
    """Preserve JSON from provider encoding through tmux/PTY and child stdin."""
    encoded = _exact_encoded_record(record_bytes, orchestration_type)
    expected = encoded.encode("utf-8") + b"\n"
    received = _round_trip_through_tmux(tmp_path, encoded)

    assert len(received) == record_bytes + 1
    assert hashlib.sha256(received).digest() == hashlib.sha256(expected).digest()
    assert received == expected
    assert json.loads(received) == json.loads(encoded)
