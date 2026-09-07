"""Tests for the Claude stream-json launch/capture boundary."""

from __future__ import annotations

import io
import json
import os
import subprocess
import termios
from unittest.mock import MagicMock, call, patch

import pytest

from cli_agent_orchestrator.models.provider_completion import ProviderCompletionInvalidError
from cli_agent_orchestrator.services import claude_completion_launcher as launcher

TERMINAL_ID = "c1a0de01"
COMPLETION_ID = "1234567890abcdef1234567890abcdef"
ARGV = [
    "--terminal-id",
    TERMINAL_ID,
    "--completion-id",
    COMPLETION_ID,
    "--",
    "claude",
    "--print",
]


class _Child:
    def __init__(self, lines: list[bytes], return_code: int = 0) -> None:
        self.stdout = iter(lines)
        self.stdin = io.BytesIO()
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.return_code if self.terminated else None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.terminated = True


def test_capture_only_treats_structured_result_as_completion() -> None:
    assistant = json.dumps({"type": "assistant", "message": {"content": "prompt answer"}})
    result = json.dumps({"type": "result", "result": "actual answer"})
    retained = MagicMock()

    with patch.object(launcher, "ingest_claude_completion", return_value=retained) as ingest:
        assert launcher._capture_result_line(TERMINAL_ID, COMPLETION_ID, b"diagnostic\n") is False
        assert (
            launcher._capture_result_line(TERMINAL_ID, COMPLETION_ID, (assistant + "\n").encode())
            is False
        )
        assert (
            launcher._capture_result_line(TERMINAL_ID, COMPLETION_ID, (result + "\n").encode())
            is True
        )

    ingest.assert_called_once_with(TERMINAL_ID, COMPLETION_ID, json.loads(result))


def test_unrelated_result_is_forwardable_but_not_correlated() -> None:
    raw = b'{"type":"result","user_message_uuid":"other"}\n'
    with patch.object(launcher, "ingest_claude_completion", return_value=None):
        assert launcher._capture_result_line(TERMINAL_ID, COMPLETION_ID, raw) is False


def test_later_result_is_a_session_lifecycle_boundary_without_claiming_callback() -> None:
    session_id = launcher.claude_session_id(TERMINAL_ID, COMPLETION_ID)
    success = json.dumps(
        {
            "type": "result",
            "session_id": session_id,
            "user_message_uuid": "99999999-8888-4777-8666-555555555555",
            "subtype": "success",
            "is_error": False,
        }
    ).encode()
    failure = success.replace(b'"success"', b'"error"')
    wrong_session = success.replace(session_id.encode(), b"0" * len(session_id))

    assert launcher._result_boundary_for_session(success, session_id) == "success"
    assert launcher._result_boundary_for_session(failure, session_id) == "error"
    assert launcher._result_boundary_for_session(wrong_session, session_id) is None


def test_initialize_control_response_requires_exact_request_and_success() -> None:
    request_id = f"cao-init-{COMPLETION_ID}"
    matching = json.dumps(
        {
            "type": "control_response",
            "response": {"subtype": "success", "request_id": request_id, "response": {}},
        }
    ).encode()
    other = matching.replace(COMPLETION_ID.encode(), b"f" * 32)
    rejected = matching.replace(b'"success"', b'"error"')

    assert launcher._is_successful_initialize_response(matching, request_id) is True
    assert launcher._is_successful_initialize_response(other, request_id) is False
    with pytest.raises(ValueError, match="rejected"):
        launcher._is_successful_initialize_response(rejected, request_id)


def test_oversized_or_invalid_utf8_line_fails_before_json_use() -> None:
    with pytest.raises(ValueError, match="size"):
        launcher._capture_result_line(
            TERMINAL_ID,
            COMPLETION_ID,
            b"x" * (launcher.MAX_REPORT_BYTES + 1),
        )
    with pytest.raises(UnicodeDecodeError):
        launcher._capture_result_line(TERMINAL_ID, COMPLETION_ID, b"\xff\n")


def test_stdin_forwarder_preserves_jsonl_bytes_and_closes_at_eof() -> None:
    child = MagicMock()
    source = io.BytesIO(b'{"type":"user","message":"one"}\nsecond\n')

    thread = launcher._start_stdin_forwarder(child, source)
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert child.stdin.write.call_args_list == [
        call(b'{"type":"user","message":"one"}\n'),
        call(b"second\n"),
    ]
    assert child.stdin.flush.call_count == 2
    child.stdin.close.assert_called_once_with()


def test_stream_stdin_mode_clears_only_canonical_input_and_restores() -> None:
    master_fd, slave_fd = os.openpty()
    source = os.fdopen(slave_fd, "rb", buffering=0)
    mode = None
    try:
        original = termios.tcgetattr(source.fileno())

        mode = launcher._stream_stdin_mode(source)

        assert mode is not None
        streaming = termios.tcgetattr(source.fileno())
        assert streaming[3] & termios.ICANON == 0
        assert streaming[3] & termios.ISIG == original[3] & termios.ISIG
        assert streaming[3] & termios.ECHO == original[3] & termios.ECHO
        assert streaming[0] == original[0]
        assert streaming[1] == original[1]
        assert streaming[2] == original[2]
        assert streaming[6][termios.VMIN] in (1, b"\x01")
        assert streaming[6][termios.VTIME] in (0, b"\x00")
        assert mode.eof_byte == bytes(original[6][termios.VEOF])

        launcher._restore_stdin_mode(mode)
        mode = None
        assert termios.tcgetattr(source.fileno()) == original
    finally:
        launcher._restore_stdin_mode(mode)
        source.close()
        os.close(master_fd)


def test_noncanonical_forwarder_preserves_configured_ctrl_d_eof() -> None:
    master_fd, slave_fd = os.openpty()
    sink_read_fd, sink_write_fd = os.pipe()
    source = os.fdopen(slave_fd, "rb", buffering=0)
    sink = os.fdopen(sink_write_fd, "wb", buffering=0)
    child = MagicMock(stdin=sink)
    thread = None
    try:
        thread = launcher._start_stdin_forwarder(child, source)
        os.write(master_fd, b'{"type":"user"}\x04must-not-follow')
        thread.join(timeout=2)

        assert thread.is_alive() is False
        assert os.read(sink_read_fd, 4096) == b'{"type":"user"}'
        assert termios.tcgetattr(source.fileno())[3] & termios.ICANON
    finally:
        if thread is not None and thread.is_alive():
            thread.stop()
        source.close()
        sink.close()
        os.close(master_fd)
        os.close(sink_read_fd)


def test_forwarder_start_failure_restores_original_terminal_mode() -> None:
    master_fd, slave_fd = os.openpty()
    source = os.fdopen(slave_fd, "rb", buffering=0)
    child = MagicMock()
    try:
        original = termios.tcgetattr(source.fileno())
        with (
            patch.object(launcher._StdinForwarder, "start", side_effect=RuntimeError("start")),
            pytest.raises(RuntimeError, match="start"),
        ):
            launcher._start_stdin_forwarder(child, source)

        assert termios.tcgetattr(source.fileno()) == original
    finally:
        source.close()
        os.close(master_fd)


def test_main_persists_correlated_result_before_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [b'{"type":"system"}\n', b'{"type":"result"}\n']
    child = _Child(lines)
    forwarded: list[bytes] = []
    order: list[str] = []
    monkeypatch.setenv("CLAUDECODE", "nested-parent")

    def capture(*_args) -> bool:
        order.append("capture")
        return len(order) == 3

    def forward(raw: bytes) -> None:
        order.append("forward")
        forwarded.append(raw)

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child) as popen,
        patch.object(launcher, "_capture_result_line", side_effect=capture),
        patch.object(launcher, "_result_boundary_for_session", return_value=None),
        patch.object(launcher, "_is_successful_initialize_response", return_value=False),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forward),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == 0

    assert order == ["capture", "forward", "capture", "forward", "forward"]
    assert forwarded == [
        *lines,
        (launcher.ADAPTER_COMPLETION_MARKER + "\n").encode("ascii"),
    ]
    child_env = popen.call_args.kwargs["env"]
    assert "CLAUDECODE" not in child_env
    assert child_env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["stdin"] == subprocess.PIPE
    child.stdin.seek(0)
    initialize = json.loads(child.stdin.readline())
    assert initialize == {
        "type": "control_request",
        "request_id": f"cao-init-{COMPLETION_ID}",
        "request": {"subtype": "initialize"},
    }


def test_main_surfaces_terminal_configuration_failure_as_adapter_error() -> None:
    child = _Child([])
    forwarded: list[bytes] = []

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(
            launcher,
            "_start_stdin_forwarder",
            side_effect=termios.error("cannot configure PTY"),
        ),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == 1

    assert forwarded == [(launcher.ADAPTER_ERROR_MARKER + "\n").encode("ascii")]
    assert child.terminated is True


def test_main_consumes_initialize_metadata_and_emits_only_ready_marker() -> None:
    control_response = b'{"type":"control_response","response":{"account":"private"}}\n'
    child = _Child([control_response, control_response])
    forwarded: list[bytes] = []

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(launcher, "_is_successful_initialize_response", return_value=True),
        patch.object(launcher, "_capture_result_line", return_value=False),
        patch.object(launcher, "_result_boundary_for_session", return_value=None),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == 1

    assert control_response not in forwarded
    assert forwarded == [
        (launcher.ADAPTER_READY_MARKER + "\n").encode("ascii"),
        (launcher.ADAPTER_ERROR_MARKER + "\n").encode("ascii"),
    ]


def test_correlated_invalid_result_is_not_forwarded_and_child_is_stopped() -> None:
    rejected = b'{"type":"result","result":"must not escape"}\n'
    child = _Child([rejected])
    forwarded: list[bytes] = []

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(
            launcher,
            "_capture_result_line",
            side_effect=ProviderCompletionInvalidError("rejected"),
        ),
        patch.object(launcher, "_is_successful_initialize_response", return_value=False),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == 1

    assert forwarded == [(launcher.ADAPTER_ERROR_MARKER + "\n").encode("ascii")]
    assert child.terminated is True


@pytest.mark.parametrize("return_code", (0, 7))
def test_eof_without_correlated_result_emits_error_boundary(return_code: int) -> None:
    child = _Child([b'{"type":"system","subtype":"init"}\n'], return_code=return_code)
    forwarded: list[bytes] = []

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(launcher, "_capture_result_line", return_value=False),
        patch.object(launcher, "_result_boundary_for_session", return_value=None),
        patch.object(launcher, "_is_successful_initialize_response", return_value=False),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == (return_code or 1)

    assert forwarded[-1] == (launcher.ADAPTER_ERROR_MARKER + "\n").encode("ascii")
    assert len(forwarded) == 2


def test_main_emits_compact_completion_boundary_for_later_turn() -> None:
    initial_line = b'{"type":"result","session_id":"ignored","subtype":"success"}\n'
    later_line = b'{"type":"result","session_id":"ignored","subtype":"success"}\n'
    child = _Child([initial_line, later_line])
    forwarded: list[bytes] = []

    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(launcher, "_is_successful_initialize_response", return_value=False),
        patch.object(launcher, "_capture_result_line", side_effect=[True, False]),
        patch.object(
            launcher,
            "_result_boundary_for_session",
            side_effect=["success", "success"],
        ),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
        patch.object(launcher.signal, "getsignal", return_value=object()),
        patch.object(launcher.signal, "signal"),
    ):
        assert launcher.main(ARGV) == 0

    assert forwarded == [
        initial_line,
        (launcher.ADAPTER_COMPLETION_MARKER + "\n").encode("ascii"),
        later_line,
        (launcher.ADAPTER_TURN_COMPLETION_MARKER + "\n").encode("ascii"),
    ]


def test_stop_child_escalates_only_after_bounded_terminate_timeout() -> None:
    child = _Child([], return_code=9)
    child.wait = MagicMock(side_effect=[subprocess.TimeoutExpired(cmd="claude", timeout=5), 9])

    launcher._stop_child(child)

    assert child.terminated is True
    assert child.killed is True
    assert child.wait.call_count == 2


def test_nonzero_exit_after_success_is_a_followup_error() -> None:
    child = _Child([b'{"type":"result"}\n'], return_code=2)
    forwarded = []
    with (
        patch.object(launcher.subprocess, "Popen", return_value=child),
        patch.object(launcher, "_is_successful_initialize_response", return_value=False),
        patch.object(launcher, "_capture_result_line", return_value=True),
        patch.object(launcher, "_result_boundary_for_session", return_value="success"),
        patch.object(launcher, "_start_stdin_forwarder"),
        patch.object(launcher, "_write_stdout", side_effect=forwarded.append),
    ):
        assert launcher.main(ARGV) == 2
    assert forwarded[-2:] == [
        (launcher.ADAPTER_COMPLETION_MARKER + "\n").encode(),
        (launcher.ADAPTER_TURN_ERROR_MARKER + "\n").encode(),
    ]


def test_stdin_pipe_relay_can_stop_without_eof() -> None:
    import os
    import time

    source_r, source_w = os.pipe()
    sink_r, sink_w = os.pipe()
    source = os.fdopen(source_r, "rb")
    sink = os.fdopen(sink_w, "wb")
    child = MagicMock(stdin=sink)
    try:
        thread = launcher._start_stdin_forwarder(child, source)
        raw = b'{"type":"user","message":"still waiting"}\n'
        os.write(source_w, raw)
        assert os.read(sink_r, len(raw)) == raw
        started = time.monotonic()
        thread.stop()
        assert not thread.is_alive()
        assert not thread.daemon
        assert time.monotonic() - started < 1.5
    finally:
        source.close()
        sink.close()
        os.close(source_w)
        os.close(sink_r)


def test_real_launcher_process_exits_after_child_crash_with_stdin_open(tmp_path):
    import os
    import sys

    fake = tmp_path / "claude"
    fake.write_text(
        "#!"
        + sys.executable
        + '\nimport sys\nprint(\'{"type":"result"}\', flush=True)\nsys.exit(2)\n'
    )
    fake.chmod(0o700)
    script = (
        "from cli_agent_orchestrator.services import claude_completion_launcher as m; "
        "m.ingest_claude_completion=lambda *args: object(); "
        "raise SystemExit(m.main(" + repr(ARGV[:-2] + [str(fake)]) + "))"
    )
    env = dict(os.environ, CAO_HOME_DIR=str(tmp_path / "isolated-cao"))
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Intentionally keep stdin open: old buffered daemon reader aborts at shutdown.
        assert process.wait(timeout=8) == 2
        stdout, stderr = process.communicate(timeout=2)
        assert (launcher.ADAPTER_TURN_ERROR_MARKER + "\n").encode() in stdout
        assert b"Fatal Python error" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
