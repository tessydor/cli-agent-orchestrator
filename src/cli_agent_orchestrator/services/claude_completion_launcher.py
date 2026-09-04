"""Launch assigned Claude Code workers on the authoritative JSONL interface.

The ordinary Claude provider remains an interactive TUI.  An assigned worker
has a stronger requirement: its final callback must come from a structured
provider result, never from rendered terminal history.  This launcher therefore
runs Claude Code with its supported ``--print --input-format stream-json
--output-format stream-json`` interface and atomically retains the matching
ResultMessage *before* forwarding that line to CAO's terminal-status pipeline.

The launcher forwards the pane's input into a real child stdin pipe, performs
the Agent SDK initialize control handshake, and mediates stdout. The matching
control response is consumed rather than logged because it may contain account
and capability metadata; provider event records otherwise remain visible.
stderr remains on the pane for normal diagnostics, and no shell is involved in
child launch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import BinaryIO, cast

from cli_agent_orchestrator.models.provider_completion import ProviderCompletionError
from cli_agent_orchestrator.services.provider_completion_report import (
    MAX_REPORT_BYTES,
    ingest_claude_completion,
)

logger = logging.getLogger(__name__)

# A line owned by CAO, not Claude.  It contains no provider or assignment text.
# ClaudeCodeProvider recognizes it only as an ERROR status boundary after this
# launcher has rejected a malformed/cross-correlated ResultMessage.
ADAPTER_ERROR_MARKER = "CAO_CLAUDE_COMPLETION_ADAPTER_ERROR_V1"
# Emitted only after a matching ResultMessage has been durably retained. This
# small lifecycle edge remains visible even when the provider's JSON result line
# is larger than StatusMonitor's rolling buffer. It never carries response text.
ADAPTER_COMPLETION_MARKER = "CAO_CLAUDE_COMPLETION_RETAINED_V1"
# Claude's stream-json process does not publish ``system/init`` until it has
# input. The Agent SDK solves that bootstrap with its documented initialize
# control request. This marker is emitted only after the matching successful
# control_response, proving that the piped transport is ready for a user record.
ADAPTER_READY_MARKER = "CAO_CLAUDE_STREAM_READY_V1"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Claude Code with authoritative completion capture"
    )
    parser.add_argument("--terminal-id", required=True)
    parser.add_argument("--completion-id", required=True)
    parser.add_argument("claude_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.claude_argv and args.claude_argv[0] == "--":
        args.claude_argv = args.claude_argv[1:]
    if not args.claude_argv or os.path.basename(args.claude_argv[0]) != "claude":
        parser.error("the wrapped command must begin with claude")
    if any("\x00" in value for value in args.claude_argv):
        parser.error("the wrapped command contains an invalid NUL byte")
    return args


def _write_stdout(raw_line: bytes) -> None:
    """Forward exact Claude stdout bytes and make status observation immediate."""
    raw_stream = getattr(sys.stdout, "buffer", None)
    if raw_stream is None:  # pragma: no cover - text-only embedded stdout
        sys.stdout.write(raw_line.decode("utf-8", errors="strict"))
        sys.stdout.flush()
        return
    stream = cast(BinaryIO, raw_stream)
    stream.write(raw_line)
    stream.flush()


def _capture_result_line(terminal_id: str, completion_id: str, raw_line: bytes) -> bool:
    """Retain a matching ResultMessage, or pass an unrelated structured line.

    The size and UTF-8 checks happen before JSON decoding.  A non-result record
    is valid stream traffic.  A result for another SDK input UUID returns from
    ``ingest_claude_completion`` without writing and remains ordinary stream
    traffic; it can never satisfy this assignment.
    """
    if len(raw_line) > MAX_REPORT_BYTES:
        raise ValueError("Claude structured output line exceeds completion size limit")
    decoded = raw_line.decode("utf-8", errors="strict")
    try:
        message = json.loads(decoded)
    except json.JSONDecodeError:
        # Claude may place non-JSON diagnostics on stdout around startup.  They
        # are display-only and never interpreted as completion evidence.
        return False
    if isinstance(message, dict) and message.get("type") == "result":
        return ingest_claude_completion(terminal_id, completion_id, message) is not None
    return False


def _is_successful_initialize_response(raw_line: bytes, request_id: str) -> bool:
    """Recognize only the response to this launcher's initialize request."""
    if len(raw_line) > MAX_REPORT_BYTES:
        raise ValueError("Claude structured output line exceeds completion size limit")
    decoded = raw_line.decode("utf-8", errors="strict")
    try:
        message = json.loads(decoded)
    except json.JSONDecodeError:
        return False
    if not isinstance(message, dict) or message.get("type") != "control_response":
        return False
    response = message.get("response")
    if not isinstance(response, dict) or response.get("request_id") != request_id:
        return False
    if response.get("subtype") != "success" or not isinstance(response.get("response"), dict):
        raise ValueError("Claude initialize control request was rejected")
    return True


def _write_initialize_request(child: subprocess.Popen[bytes], request_id: str) -> None:
    """Send the minimal Agent SDK initialize control request to Claude."""
    if child.stdin is None:  # pragma: no cover - guaranteed by stdin=PIPE
        raise OSError("Claude stdin pipe was not created")
    payload = {
        "type": "control_request",
        "request_id": request_id,
        "request": {"subtype": "initialize"},
    }
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    child.stdin.write(raw)
    child.stdin.flush()


def _start_stdin_forwarder(
    child: subprocess.Popen[bytes], source: BinaryIO | None = None
) -> threading.Thread:
    """Forward terminal JSONL into Claude's non-TTY stdin pipe.

    A daemon thread is intentional: the main thread owns child/output lifetime,
    while a terminal ``readline`` can remain blocked after the child exits.
    No input bytes are logged or interpreted here.
    """
    sink = child.stdin
    if sink is None:  # pragma: no cover - guaranteed by stdin=PIPE
        raise OSError("Claude stdin pipe was not created")
    if source is None:
        raw_source = getattr(sys.stdin, "buffer", None)
        if raw_source is None:  # pragma: no cover - terminal stdin always buffered
            raise OSError("launcher stdin has no binary stream")
        source = cast(BinaryIO, raw_source)

    def _forward() -> None:
        try:
            while True:
                raw_line = source.readline()
                if not raw_line:
                    sink.close()
                    return
                if isinstance(raw_line, str):  # pragma: no cover - embedded text stdin
                    raw_line = raw_line.encode("utf-8", errors="strict")
                sink.write(raw_line)
                sink.flush()
        except (BrokenPipeError, OSError, ValueError):
            # The main stdout loop observes child exit and publishes the
            # adapter error boundary when no correlated ResultMessage exists.
            logger.exception("Claude structured stdin forwarding stopped")

    thread = threading.Thread(
        target=_forward,
        name="cao-claude-stdin-forwarder",
        daemon=True,
    )
    thread.start()
    return thread


def _stop_child(child: subprocess.Popen[bytes]) -> None:
    """Bound cleanup after rejecting an authoritative-result candidate."""
    if child.stdin is not None:
        try:
            child.stdin.close()
        except OSError:
            pass
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Claude, retaining each correlated ResultMessage before publication."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Ctrl-C is delivered to the whole foreground process group.  Claude owns
    # the SDK cancellation and emits terminal_reason=aborted_streaming/tools;
    # the parent must stay alive long enough to retain that ResultMessage.
    previous_sigint = signal.getsignal(signal.SIGINT)
    child: subprocess.Popen[bytes] | None = None
    try:
        child_env = dict(os.environ)
        child_env.pop("CLAUDECODE", None)
        child_env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
        child = subprocess.Popen(
            args.claude_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            shell=False,
            env=child_env,
        )
        # Set this only after Popen so Claude inherits the original SIGINT
        # disposition and can turn Ctrl-C into a structured cancelled result.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise OSError("Claude stdout pipe was not created")

        initialize_request_id = f"cao-init-{args.completion_id}"
        # Match ClaudeAgentSDKClient's supported bootstrap ordering: initialize
        # first, then allow user records onto the same pipe.
        _write_initialize_request(child, initialize_request_id)
        _start_stdin_forwarder(child)

        saw_correlated_result = False
        ready_published = False
        for raw_line in child.stdout:
            try:
                initialize_ready = _is_successful_initialize_response(
                    raw_line, initialize_request_id
                )
                correlated_result = _capture_result_line(
                    args.terminal_id, args.completion_id, raw_line
                )
                saw_correlated_result = correlated_result or saw_correlated_result
            except (ProviderCompletionError, UnicodeError, OSError, ValueError):
                # Never echo the rejected ResultMessage: status processing must
                # not observe it as a successful turn.  Emit only a constant
                # adapter-owned error boundary and stop this worker.
                logger.exception("Authoritative Claude completion ingestion failed")
                _write_stdout((ADAPTER_ERROR_MARKER + "\n").encode("ascii"))
                _stop_child(child)
                return 1
            if initialize_ready:
                # Consume the control response rather than retaining account,
                # command, model, and capability metadata in terminal logs.
                if not ready_published:
                    ready_published = True
                    _write_stdout((ADAPTER_READY_MARKER + "\n").encode("ascii"))
                continue
            _write_stdout(raw_line)
            if correlated_result:
                # The report already exists at this point. Publishing a compact
                # adapter-owned edge after the potentially multi-megabyte JSON
                # record makes completion detection independent of buffer size.
                _write_stdout((ADAPTER_COMPLETION_MARKER + "\n").encode("ascii"))

        return_code = child.wait()
        if saw_correlated_result:
            return return_code

        # A supported assigned-worker turn always terminates with exactly one
        # ResultMessage. EOF without a correlated result is an authoritative
        # adapter failure regardless of the child's numeric exit status.
        logger.error("Claude stream ended without a correlated ResultMessage")
        _write_stdout((ADAPTER_ERROR_MARKER + "\n").encode("ascii"))
        return return_code if return_code != 0 else 1
    except (OSError, ValueError):
        logger.exception("Cannot launch Claude assigned worker completion adapter")
        _write_stdout((ADAPTER_ERROR_MARKER + "\n").encode("ascii"))
        if child is not None:
            _stop_child(child)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":  # pragma: no cover - exercised by live assigned workers
    raise SystemExit(main())
