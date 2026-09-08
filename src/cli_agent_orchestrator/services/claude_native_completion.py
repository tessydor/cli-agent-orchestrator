"""Native Claude hooks: capture a correlated turn without controlling permission decisions.

Requires Claude Code >=2.1.196 (prompt_id). Missing correlation fails closed.
The local hook process has the same OS trust boundary as other provider adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from cli_agent_orchestrator.models.provider_completion import (
    ProviderCompletionCorrelationError,
    ProviderCompletionInvalidError,
    ProviderCompletionReport,
    input_messages_sha256,
    utf8_sha256,
)
from cli_agent_orchestrator.utils.atomic_file import locked_atomic_rewrite, locked_atomic_write

from . import provider_completion_report as reports


def _path(terminal_id: str, completion_id: str, suffix: str) -> Path:
    dispatch, _, _ = reports._paths("claude_code", terminal_id, completion_id)
    return dispatch.with_suffix(suffix)


def configure_native(terminal_id: str, completion_id: str) -> None:
    path = _path(terminal_id, completion_id, ".native.json")
    reports._ensure_private_directory(path.parent)
    locked_atomic_write(path, json.dumps({"transport": "native", "version": 1}), file_mode=0o600)


def is_native(terminal_id: str, completion_id: Optional[str]) -> bool:
    if completion_id is None:
        return False
    path = _path(terminal_id, completion_id, ".native.json")
    if not path.exists():
        return False  # Existing assigned workers retain their streaming transport.
    if json.loads(path.read_text()) != {"transport": "native", "version": 1}:
        raise ProviderCompletionInvalidError("Invalid persisted Claude transport")
    return True


def ingest(terminal_id: str, completion_id: str, event: dict[str, Any]) -> None:
    expected = reports.claude_session_id(terminal_id, completion_id)
    if event.get("session_id") != expected:
        raise ProviderCompletionCorrelationError("Native hook session mismatch")
    if event.get("agent_id"):
        return  # Subagent output is never the parent assignment result.
    kind = event.get("hook_event_name")
    prompt_id = reports._validate_claude_uuid(
        reports._required_string(event, "prompt_id"), field="prompt_id"
    )
    binding = _path(terminal_id, completion_id, ".prompt.json")
    if kind == "UserPromptSubmit":
        digest = utf8_sha256(reports._required_string(event, "prompt"))
        bound = reports._bound_dispatch_digests("claude_code", terminal_id, completion_id)
        payload = {"prompt_id": prompt_id, "digest": digest, "matched": digest in bound}
        locked_atomic_rewrite(binding, lambda _: json.dumps(payload), file_mode=0o600)
        return
    if kind not in ("Stop", "StopFailure"):
        raise ProviderCompletionInvalidError("Unsupported native completion event")
    try:
        current = json.loads(binding.read_text())
    except FileNotFoundError:
        return  # Never manufacture a result without an observed prompt.
    if current.get("prompt_id") != prompt_id or not current.get("matched"):
        return
    digest = current["digest"]
    if digest not in reports._bound_dispatch_digests("claude_code", terminal_id, completion_id):
        raise ProviderCompletionCorrelationError("Native dispatch mismatch")
    failed = kind == "StopFailure"
    text = event.get("last_assistant_message")
    if failed and not text:
        text = "Claude API failure: " + reports._required_string(event, "error")
    if not isinstance(text, str):
        raise ProviderCompletionInvalidError("Native hook response must be text")
    if not text.strip() and not failed:
        raise ProviderCompletionInvalidError("Native Stop has no final response")
    report = ProviderCompletionReport(
        provider="claude_code",
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=expected,
        provider_turn_id=prompt_id,
        input_messages_sha256=input_messages_sha256(
            [reports.claude_input_id(terminal_id, completion_id, digest)]
        ),
        dispatched_input_sha256=digest,
        final_response=text,
        final_response_sha256=utf8_sha256(text),
        source_reference=f"provider-completion:claude_code:{expected}:{prompt_id}",
        completion_state="failure" if failed else "success",
        provider_input_id=reports.claude_input_id(terminal_id, completion_id, digest),
        provider_result_subtype="error_during_execution" if failed else "success",
        provider_terminal_reason="native_stop_failure" if failed else "completed",
        provider_is_error=failed,
    )
    reports._persist_report(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-id", required=True)
    parser.add_argument("--completion-id", required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(reports.MAX_REPORT_BYTES + 1)
    if len(raw) > reports.MAX_REPORT_BYTES:
        raise ProviderCompletionInvalidError("Native hook exceeds size limit")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise ProviderCompletionInvalidError("Native hook must be an object")
    ingest(args.terminal_id, args.completion_id, event)


if __name__ == "__main__":
    main()
