"""Durable ingestion and retrieval of authoritative provider completion data.

The module is also the process entry point used by provider-native completion
hooks and launch adapters.  Codex appends one structured JSON argument to its
configured ``agent-turn-complete`` command.  Assigned Claude Code workers run
through the supported CLI ``stream-json`` interface, whose authoritative
``ResultMessage`` is validated and retained before it is exposed to terminal
status processing.  Callback delivery later reads either source through the
provider-neutral ``BaseProvider.get_completion_report`` contract.

No terminal history, assignment prompt parsing, or response-text heuristic is
used here.  Correlation is an exact equality check between the final structured
provider input message and a digest bound immediately before CAO dispatches the
assigned task.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from cli_agent_orchestrator.constants import PROVIDER_COMPLETION_REPORT_DIR
from cli_agent_orchestrator.models.provider_completion import (
    ProviderCompletionConflictError,
    ProviderCompletionCorrelationError,
    ProviderCompletionError,
    ProviderCompletionInvalidError,
    ProviderCompletionReport,
    ProviderCompletionUnavailableError,
    input_messages_sha256,
    utf8_sha256,
)
from cli_agent_orchestrator.utils.atomic_file import locked_atomic_rewrite, locked_atomic_write

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_DISPATCH_ATTEMPTS = 16

_TERMINAL_ID_RE = re.compile(r"^[a-f0-9]{8}$")
_COMPLETION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

# Claude Code 2.1.259's installed ResultMessage union.  Keeping this closed set
# makes a future incompatible wire change fail closed instead of silently
# promoting a new terminal outcome to success.
_CLAUDE_RESULT_SUBTYPES = frozenset(
    {
        "success",
        "error_during_execution",
        "error_max_turns",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
)
_CLAUDE_CANCELLED_TERMINAL_REASONS = frozenset({"aborted_streaming", "aborted_tools"})
_COMPLETION_STATES = frozenset({"success", "failure", "cancelled", "terminated"})

# UUIDv5 identities are derived from CAO's immutable terminal/completion route
# and the exact bound dispatch digest.  They are valid Claude SDK identifiers,
# stable across server-side recovery, and cannot collide across assignments
# unless every correlated CAO identity is also identical.
_CLAUDE_COMPLETION_NAMESPACE = uuid.UUID("9f3dbf1e-ea10-4f73-8a91-496f2bb43f23")


def claude_session_id(terminal_id: str, completion_id: str) -> str:
    """Return the deterministic Claude session UUID for one CAO completion."""
    _validate_identity("claude_code", terminal_id, completion_id)
    return str(
        uuid.uuid5(
            _CLAUDE_COMPLETION_NAMESPACE,
            f"claude-session:{terminal_id}:{completion_id}",
        )
    )


def claude_input_id(terminal_id: str, completion_id: str, dispatch_sha256: str) -> str:
    """Return the Claude user-message UUID bound to exact dispatched bytes."""
    _validate_identity("claude_code", terminal_id, completion_id)
    if not _SHA256_RE.fullmatch(dispatch_sha256):
        raise ProviderCompletionInvalidError("Claude dispatch digest is malformed")
    return str(
        uuid.uuid5(
            _CLAUDE_COMPLETION_NAMESPACE,
            f"claude-input:{terminal_id}:{completion_id}:{dispatch_sha256}",
        )
    )


def _report_root() -> Path:
    """Return the patchable report root used by this process."""
    return PROVIDER_COMPLETION_REPORT_DIR


def _validate_identity(provider: str, terminal_id: str, completion_id: str) -> None:
    if not _PROVIDER_RE.fullmatch(provider):
        raise ProviderCompletionInvalidError(f"invalid provider identity: {provider!r}")
    if not _TERMINAL_ID_RE.fullmatch(terminal_id):
        raise ProviderCompletionInvalidError(f"invalid terminal identity: {terminal_id!r}")
    if not _COMPLETION_ID_RE.fullmatch(completion_id):
        raise ProviderCompletionInvalidError(f"invalid completion identity: {completion_id!r}")


def _paths(provider: str, terminal_id: str, completion_id: str) -> tuple[Path, Path, Path]:
    _validate_identity(provider, terminal_id, completion_id)
    directory = _report_root() / provider / terminal_id
    stem = directory / completion_id
    return (
        stem.with_suffix(".dispatch.json"),
        stem.with_suffix(".report.json"),
        stem.with_suffix(".conflict.json"),
    )


def _ensure_private_directory(path: Path) -> None:
    """Create the report hierarchy with owner-only directory permissions."""
    root = _report_root()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:  # defensive: all public paths come from _paths
        raise ProviderCompletionInvalidError("completion report path escaped its root") from exc

    directories = [root]
    current = root
    for component in relative.parts:
        current /= component
        directories.append(current)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            # The atomic files remain explicitly 0600. A read-only/non-owned
            # mount may reject chmod; the subsequent write surfaces a real
            # failure if the directory is not usable.
            pass


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ProviderCompletionUnavailableError(f"authoritative {label} is not available") from exc
    if len(raw) > MAX_REPORT_BYTES:
        raise ProviderCompletionInvalidError(f"authoritative {label} exceeds size limit")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderCompletionInvalidError(f"authoritative {label} is malformed") from exc
    if not isinstance(value, dict):
        raise ProviderCompletionInvalidError(f"authoritative {label} is not a JSON object")
    return value


def bind_completion_dispatch(
    provider: str,
    terminal_id: str,
    completion_id: str,
    dispatched_input: str,
) -> str:
    """Durably bind an exact task input before its external paste side effect.

    Resubmission can legitimately produce a second exact input (for example the
    first attempt includes a memory prelude and a retry does not), so a bounded
    immutable set of admissible attempt digests is retained.  No prompt text is
    stored in this correlation file.
    """
    dispatch_path, _, _ = _paths(provider, terminal_id, completion_id)
    _ensure_private_directory(dispatch_path.parent)
    digest = utf8_sha256(dispatched_input)

    def _merge(existing: str) -> str:
        if existing:
            try:
                payload = json.loads(existing)
            except json.JSONDecodeError as exc:
                raise ProviderCompletionInvalidError(
                    "authoritative dispatch correlation is malformed"
                ) from exc
            if not isinstance(payload, dict):
                raise ProviderCompletionInvalidError(
                    "authoritative dispatch correlation is not a JSON object"
                )
            expected_identity = (provider, terminal_id, completion_id)
            actual_identity = (
                payload.get("provider"),
                payload.get("terminal_id"),
                payload.get("completion_id"),
            )
            if (
                actual_identity != expected_identity
                or payload.get("schema_version") != SCHEMA_VERSION
            ):
                raise ProviderCompletionCorrelationError(
                    "persisted dispatch correlation has a mismatched identity"
                )
            digests = payload.get("dispatched_input_sha256")
            if not isinstance(digests, list) or not all(
                isinstance(item, str) and _SHA256_RE.fullmatch(item) for item in digests
            ):
                raise ProviderCompletionInvalidError(
                    "persisted dispatch correlation has invalid input digests"
                )
        else:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "provider": provider,
                "terminal_id": terminal_id,
                "completion_id": completion_id,
                "dispatched_input_sha256": [],
            }
            digests = payload["dispatched_input_sha256"]

        if digest not in digests:
            if len(digests) >= MAX_DISPATCH_ATTEMPTS:
                raise ProviderCompletionInvalidError(
                    "too many distinct dispatch attempts for one completion"
                )
            digests.append(digest)
        return _canonical_json(payload)

    locked_atomic_rewrite(dispatch_path, _merge, file_mode=0o600)
    return digest


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ProviderCompletionInvalidError(f"provider completion field {key!r} is not text")
    return value


def _validate_provider_id(value: str, *, field: str) -> str:
    if not _PROVIDER_ID_RE.fullmatch(value):
        raise ProviderCompletionInvalidError(f"provider completion field {field!r} is invalid")
    return value


def _validate_claude_uuid(value: str, *, field: str) -> str:
    """Validate the canonical UUID strings exposed by Claude's SDK stream."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProviderCompletionInvalidError(
            f"Claude completion field {field!r} is not a UUID"
        ) from exc
    if str(parsed) != value:
        raise ProviderCompletionInvalidError(
            f"Claude completion field {field!r} is not a canonical UUID"
        )
    return value


def _classify_claude_outcome(
    *,
    subtype: str,
    is_error: bool,
    terminal_reason: str | None,
    result: str,
) -> str:
    """Map Claude 2.1.259 ResultMessage fields to CAO's terminal outcome."""
    if subtype == "success" and is_error is False and terminal_reason == "completed":
        if not result.strip():
            raise ProviderCompletionInvalidError(
                "Claude completion has no authoritative non-empty assistant result"
            )
        return "success"
    if terminal_reason in _CLAUDE_CANCELLED_TERMINAL_REASONS:
        return "cancelled"
    if subtype != "success" or is_error is True:
        return "failure"
    # Non-error ResultMessages such as max-turn/tool-deferred/background
    # boundaries are terminal for this assigned turn but are not successful
    # final answers and are not cancellations.
    return "terminated"


def _bound_dispatch_digests(
    provider: str,
    terminal_id: str,
    completion_id: str,
) -> list[str]:
    """Load the immutable exact-input digests for one completion identity."""
    dispatch_path, _, _ = _paths(provider, terminal_id, completion_id)
    payload = _load_json_object(dispatch_path, label="dispatch correlation")
    expected_identity = (SCHEMA_VERSION, provider, terminal_id, completion_id)
    actual_identity = (
        payload.get("schema_version"),
        payload.get("provider"),
        payload.get("terminal_id"),
        payload.get("completion_id"),
    )
    if actual_identity != expected_identity:
        raise ProviderCompletionCorrelationError(
            "authoritative dispatch identity does not match the assigned completion"
        )
    digests = payload.get("dispatched_input_sha256")
    if (
        not isinstance(digests, list)
        or not digests
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in digests)
    ):
        raise ProviderCompletionInvalidError("dispatch correlation digests are malformed")
    return digests


def _match_claude_dispatch(
    terminal_id: str,
    completion_id: str,
    result_message: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Match a Claude result input UUID to exactly one bound dispatch digest.

    ``None`` means this is a different turn in the same long-lived stream.  A
    result claiming the assigned input plus any additional batched/queued input
    is ambiguous and is rejected: its final text no longer answers only the
    exact dispatched assignment.
    """
    user_message_uuid = result_message.get("user_message_uuid")
    if user_message_uuid is None:
        return None
    if not isinstance(user_message_uuid, str):
        raise ProviderCompletionInvalidError("Claude result user_message_uuid is malformed")
    _validate_claude_uuid(user_message_uuid, field="user_message_uuid")

    digests = _bound_dispatch_digests("claude_code", terminal_id, completion_id)
    candidates = [
        (digest, claude_input_id(terminal_id, completion_id, digest)) for digest in digests
    ]
    matches = [candidate for candidate in candidates if candidate[1] == user_message_uuid]
    if not matches:
        return None
    if len(matches) != 1:
        raise ProviderCompletionCorrelationError(
            "Claude result input identity ambiguously matches dispatch attempts"
        )

    user_message_uuids = result_message.get("user_message_uuids")
    if user_message_uuids is not None:
        if not isinstance(user_message_uuids, list) or not all(
            isinstance(value, str) for value in user_message_uuids
        ):
            raise ProviderCompletionInvalidError("Claude result user_message_uuids is malformed")
        for value in user_message_uuids:
            _validate_claude_uuid(value, field="user_message_uuids")
        if user_message_uuids != [user_message_uuid]:
            raise ProviderCompletionCorrelationError(
                "Claude result combined the assigned input with another user message"
            )
    return matches[0]


def ingest_claude_completion(
    terminal_id: str,
    completion_id: str,
    result_message: Mapping[str, Any],
) -> ProviderCompletionReport | None:
    """Validate and retain one Claude Code 2.1.259 structured ResultMessage.

    A result for another input in the same streaming session returns ``None``;
    it is not evidence about this assignment.  Once the assigned input UUID is
    claimed, every session, batch, result, outcome, and response field is
    validated before an immutable report is written.
    """
    provider = "claude_code"
    _validate_identity(provider, terminal_id, completion_id)
    if result_message.get("type") != "result":
        raise ProviderCompletionInvalidError("Claude completion is not a ResultMessage")

    provider_session_id = _validate_claude_uuid(
        _required_string(result_message, "session_id"), field="session_id"
    )
    provider_turn_id = _validate_claude_uuid(_required_string(result_message, "uuid"), field="uuid")
    provider_result_subtype = _required_string(result_message, "subtype")
    if provider_result_subtype not in _CLAUDE_RESULT_SUBTYPES:
        raise ProviderCompletionInvalidError("Claude result subtype is unsupported")
    is_error = result_message.get("is_error")
    if not isinstance(is_error, bool):
        raise ProviderCompletionInvalidError("Claude result is_error is malformed")

    provider_terminal_reason = result_message.get("terminal_reason")
    if provider_terminal_reason is not None:
        if not isinstance(provider_terminal_reason, str):
            raise ProviderCompletionInvalidError("Claude terminal_reason is malformed")
        _validate_provider_id(provider_terminal_reason, field="terminal_reason")

    matched = _match_claude_dispatch(terminal_id, completion_id, result_message)
    if matched is None:
        return None
    dispatched_input_sha256, provider_input_id = matched

    expected_session_id = claude_session_id(terminal_id, completion_id)
    if provider_session_id != expected_session_id:
        raise ProviderCompletionCorrelationError(
            "Claude result session does not match the dispatched worker session"
        )

    raw_result = result_message.get("result", "")
    if not isinstance(raw_result, str):
        raise ProviderCompletionInvalidError("Claude result text is malformed")

    completion_state = _classify_claude_outcome(
        subtype=provider_result_subtype,
        is_error=is_error,
        terminal_reason=provider_terminal_reason,
        result=raw_result,
    )

    report = ProviderCompletionReport(
        provider=provider,
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=provider_session_id,
        provider_turn_id=provider_turn_id,
        input_messages_sha256=input_messages_sha256([provider_input_id]),
        dispatched_input_sha256=dispatched_input_sha256,
        final_response=raw_result,
        final_response_sha256=utf8_sha256(raw_result),
        source_reference=(
            f"provider-completion:{provider}:{provider_session_id}:{provider_turn_id}"
        ),
        completion_state=completion_state,
        provider_input_id=provider_input_id,
        provider_result_subtype=provider_result_subtype,
        provider_terminal_reason=provider_terminal_reason,
        provider_is_error=is_error,
    )
    _persist_report(report)
    return report


def ingest_codex_completion(
    terminal_id: str,
    completion_id: str,
    notification: Mapping[str, Any],
) -> ProviderCompletionReport:
    """Validate and retain one Codex native ``agent-turn-complete`` payload."""
    provider = "codex"
    _validate_identity(provider, terminal_id, completion_id)
    if notification.get("type") != "agent-turn-complete":
        raise ProviderCompletionInvalidError("Codex notification is not agent-turn-complete")

    provider_session_id = _validate_provider_id(
        _required_string(notification, "thread-id"), field="thread-id"
    )
    provider_turn_id = _validate_provider_id(
        _required_string(notification, "turn-id"), field="turn-id"
    )
    raw_messages = notification.get("input-messages")
    if (
        not isinstance(raw_messages, list)
        or not raw_messages
        or not all(isinstance(message, str) for message in raw_messages)
    ):
        raise ProviderCompletionInvalidError(
            "Codex completion input-messages must be a non-empty text array"
        )
    input_messages: Sequence[str] = raw_messages
    final_response = _required_string(notification, "last-assistant-message")

    # A whitespace-only response is not a successful callback, but accepted
    # responses are never stripped or normalized: the digest below covers the
    # exact UTF-8 bytes supplied by Codex.
    if not final_response.strip():
        raise ProviderCompletionInvalidError(
            "Codex completion has no authoritative non-empty assistant response"
        )

    report = ProviderCompletionReport(
        provider=provider,
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=provider_session_id,
        provider_turn_id=provider_turn_id,
        input_messages_sha256=input_messages_sha256(input_messages),
        dispatched_input_sha256=utf8_sha256(input_messages[-1]),
        final_response=final_response,
        final_response_sha256=utf8_sha256(final_response),
        source_reference=(
            f"provider-completion:{provider}:{provider_session_id}:{provider_turn_id}"
        ),
    )
    _persist_report(report)
    return report


def ingest_mock_completion(
    terminal_id: str,
    completion_id: str,
    provider_turn_id: str,
    dispatched_input: str,
    final_response: str,
) -> ProviderCompletionReport:
    """Test-only structured provider boundary used by the mock CLI binary."""
    return _ingest_test_completion(
        provider="mock_cli",
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=f"mock-{terminal_id}",
        provider_turn_id=provider_turn_id,
        input_messages=[dispatched_input],
        final_response=final_response,
    )


def _ingest_test_completion(
    *,
    provider: str,
    terminal_id: str,
    completion_id: str,
    provider_session_id: str,
    provider_turn_id: str,
    input_messages: Sequence[str],
    final_response: str,
) -> ProviderCompletionReport:
    """Build a report for deterministic adapter tests without terminal parsing."""
    _validate_identity(provider, terminal_id, completion_id)
    _validate_provider_id(provider_session_id, field="provider_session_id")
    _validate_provider_id(provider_turn_id, field="provider_turn_id")
    if not input_messages or not all(isinstance(value, str) for value in input_messages):
        raise ProviderCompletionInvalidError("mock completion input_messages are invalid")
    if not isinstance(final_response, str) or not final_response.strip():
        raise ProviderCompletionInvalidError("mock completion final_response is empty")
    report = ProviderCompletionReport(
        provider=provider,
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=provider_session_id,
        provider_turn_id=provider_turn_id,
        input_messages_sha256=input_messages_sha256(input_messages),
        dispatched_input_sha256=utf8_sha256(input_messages[-1]),
        final_response=final_response,
        final_response_sha256=utf8_sha256(final_response),
        source_reference=(
            f"provider-completion:{provider}:{provider_session_id}:{provider_turn_id}"
        ),
    )
    _persist_report(report)
    return report


def _report_payload(report: ProviderCompletionReport) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provider": report.provider,
        "terminal_id": report.terminal_id,
        "completion_id": report.completion_id,
        "provider_session_id": report.provider_session_id,
        "provider_turn_id": report.provider_turn_id,
        "input_messages_sha256": report.input_messages_sha256,
        "dispatched_input_sha256": report.dispatched_input_sha256,
        "final_response": report.final_response,
        "final_response_sha256": report.final_response_sha256,
        "source_reference": report.source_reference,
    }
    if report.provider == "claude_code":
        # Additive fields are written only for Claude reports.  Re-emitting an
        # old Codex notification must remain byte-identical to a pre-adapter V1
        # report so immutable duplicate handling does not manufacture a conflict.
        payload.update(
            {
                "completion_state": report.completion_state,
                "provider_input_id": report.provider_input_id,
                "provider_result_subtype": report.provider_result_subtype,
                "provider_terminal_reason": report.provider_terminal_reason,
                "provider_is_error": report.provider_is_error,
            }
        )
    return payload


def _persist_report(report: ProviderCompletionReport) -> None:
    _, report_path, conflict_path = _paths(
        report.provider, report.terminal_id, report.completion_id
    )
    _ensure_private_directory(report_path.parent)
    new_content = _canonical_json(_report_payload(report))
    conflict: list[str] = []

    def _preserve_first(existing: str) -> str:
        if not existing or existing == new_content:
            return new_content
        conflict.append(utf8_sha256(existing))
        return existing

    locked_atomic_rewrite(report_path, _preserve_first, file_mode=0o600)
    if not conflict:
        return

    # Never overwrite the first provider report.  Retain only digests for a
    # conflict so both the immutable evidence and the fact of disagreement
    # survive restart without duplicating response text.
    conflict_payload = _canonical_json(
        {
            "schema_version": SCHEMA_VERSION,
            "provider": report.provider,
            "terminal_id": report.terminal_id,
            "completion_id": report.completion_id,
            "first_report_sha256": conflict[0],
            "conflicting_report_sha256": utf8_sha256(new_content),
        }
    )
    try:
        locked_atomic_write(
            conflict_path,
            conflict_payload,
            overwrite=False,
            file_mode=0o600,
        )
    except FileExistsError:
        pass
    raise ProviderCompletionConflictError(
        "conflicting authoritative reports claim the same completion"
    )


def load_completion_report(
    provider: str,
    terminal_id: str,
    completion_id: str,
) -> ProviderCompletionReport:
    """Load, validate, and exactly correlate one retained report."""
    dispatch_path, report_path, conflict_path = _paths(provider, terminal_id, completion_id)
    if conflict_path.exists():
        raise ProviderCompletionConflictError(
            "conflicting authoritative reports claim the same completion"
        )
    report_payload = _load_json_object(report_path, label="provider completion report")
    dispatch_payload = _load_json_object(dispatch_path, label="dispatch correlation")

    expected_identity = (SCHEMA_VERSION, provider, terminal_id, completion_id)
    for label, payload in (("report", report_payload), ("dispatch", dispatch_payload)):
        actual_identity = (
            payload.get("schema_version"),
            payload.get("provider"),
            payload.get("terminal_id"),
            payload.get("completion_id"),
        )
        if actual_identity != expected_identity:
            raise ProviderCompletionCorrelationError(
                f"authoritative {label} identity does not match the assigned completion"
            )

    digests = dispatch_payload.get("dispatched_input_sha256")
    report_dispatch_digest = report_payload.get("dispatched_input_sha256")
    if not isinstance(digests, list) or not all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in digests
    ):
        raise ProviderCompletionInvalidError("dispatch correlation digests are malformed")
    if not isinstance(report_dispatch_digest, str) or report_dispatch_digest not in digests:
        raise ProviderCompletionCorrelationError(
            "provider completion does not match an exact dispatched task input"
        )

    required_text = (
        "provider_session_id",
        "provider_turn_id",
        "input_messages_sha256",
        "final_response",
        "final_response_sha256",
        "source_reference",
    )
    if any(not isinstance(report_payload.get(key), str) for key in required_text):
        raise ProviderCompletionInvalidError("provider completion report fields are malformed")
    if provider == "claude_code" and "completion_state" not in report_payload:
        raise ProviderCompletionInvalidError("Claude provider completion state is missing")
    completion_state = report_payload.get("completion_state", "success")
    if completion_state not in _COMPLETION_STATES:
        raise ProviderCompletionInvalidError("provider completion state is invalid")
    if provider != "claude_code" and completion_state != "success":
        raise ProviderCompletionInvalidError("legacy provider report has a non-success state")

    final_response = report_payload["final_response"]
    if completion_state == "success" and not final_response.strip():
        raise ProviderCompletionInvalidError(
            "provider completion has no authoritative non-empty assistant response"
        )
    if report_payload["final_response_sha256"] != utf8_sha256(final_response):
        raise ProviderCompletionInvalidError("provider completion response digest is invalid")
    if not _SHA256_RE.fullmatch(report_payload["input_messages_sha256"]):
        raise ProviderCompletionInvalidError("provider input-message digest is invalid")
    _validate_provider_id(report_payload["provider_session_id"], field="provider_session_id")
    _validate_provider_id(report_payload["provider_turn_id"], field="provider_turn_id")
    expected_reference = (
        f"provider-completion:{provider}:{report_payload['provider_session_id']}:"
        f"{report_payload['provider_turn_id']}"
    )
    if report_payload["source_reference"] != expected_reference:
        raise ProviderCompletionInvalidError("provider completion source reference is invalid")

    provider_input_id: str | None = None
    provider_result_subtype: str | None = None
    provider_terminal_reason: str | None = None
    provider_is_error: bool | None = None
    if provider == "claude_code":
        required_claude_fields = {
            "provider_input_id",
            "provider_result_subtype",
            "provider_terminal_reason",
            "provider_is_error",
        }
        if not required_claude_fields.issubset(report_payload):
            raise ProviderCompletionInvalidError("Claude provider report fields are missing")
        provider_input_id = report_payload.get("provider_input_id")
        provider_result_subtype = report_payload.get("provider_result_subtype")
        provider_terminal_reason = report_payload.get("provider_terminal_reason")
        provider_is_error = report_payload.get("provider_is_error")
        if not isinstance(provider_input_id, str):
            raise ProviderCompletionInvalidError("Claude provider input identity is malformed")
        _validate_claude_uuid(report_payload["provider_session_id"], field="provider_session_id")
        _validate_claude_uuid(report_payload["provider_turn_id"], field="provider_turn_id")
        _validate_claude_uuid(provider_input_id, field="provider_input_id")
        if (
            not isinstance(provider_result_subtype, str)
            or provider_result_subtype not in _CLAUDE_RESULT_SUBTYPES
        ):
            raise ProviderCompletionInvalidError("Claude result subtype is malformed")
        if provider_terminal_reason is not None:
            if not isinstance(provider_terminal_reason, str):
                raise ProviderCompletionInvalidError("Claude terminal reason is malformed")
            _validate_provider_id(provider_terminal_reason, field="provider_terminal_reason")
        if not isinstance(provider_is_error, bool):
            raise ProviderCompletionInvalidError("Claude result is_error is malformed")

        expected_state = _classify_claude_outcome(
            subtype=provider_result_subtype,
            is_error=provider_is_error,
            terminal_reason=provider_terminal_reason,
            result=final_response,
        )
        if completion_state != expected_state:
            raise ProviderCompletionInvalidError(
                "Claude completion state contradicts its ResultMessage evidence"
            )

        expected_input_ids = {
            claude_input_id(terminal_id, completion_id, digest) for digest in digests
        }
        if provider_input_id not in expected_input_ids or report_payload[
            "input_messages_sha256"
        ] != input_messages_sha256([provider_input_id]):
            raise ProviderCompletionCorrelationError(
                "Claude report input identity does not match the exact dispatched task"
            )

    return ProviderCompletionReport(
        provider=provider,
        terminal_id=terminal_id,
        completion_id=completion_id,
        provider_session_id=report_payload["provider_session_id"],
        provider_turn_id=report_payload["provider_turn_id"],
        input_messages_sha256=report_payload["input_messages_sha256"],
        dispatched_input_sha256=report_dispatch_digest,
        final_response=final_response,
        final_response_sha256=report_payload["final_response_sha256"],
        source_reference=report_payload["source_reference"],
        completion_state=completion_state,
        provider_input_id=provider_input_id,
        provider_result_subtype=provider_result_subtype,
        provider_terminal_reason=provider_terminal_reason,
        provider_is_error=provider_is_error,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retain an authoritative provider completion")
    parser.add_argument("--provider", choices=("codex", "mock_cli"), required=True)
    parser.add_argument("--terminal-id", required=True)
    parser.add_argument("--completion-id", required=True)
    parser.add_argument("--forward-notify-json")
    parser.add_argument("--turn-id")
    parser.add_argument("--input-message")
    parser.add_argument("--final-response")
    parser.add_argument("notification_json", nargs="?")
    return parser.parse_args(argv)


def _forward_codex_notification(forward_argv_json: str, notification_json: str) -> None:
    """Launch the previously effective Codex notifier without a shell.

    Codex's native notifier is fire-and-forget: it appends the notification as
    one argv element, redirects all standard streams to null, and only observes
    whether spawning succeeded.  Preserve those semantics rather than waiting
    for, interpreting, or accidentally shell-expanding the user's notifier.
    """
    if len(forward_argv_json.encode("utf-8", errors="strict")) > MAX_REPORT_BYTES:
        raise ProviderCompletionInvalidError("forwarded Codex notify argv exceeds size limit")
    try:
        raw_argv = json.loads(forward_argv_json)
    except json.JSONDecodeError as exc:
        raise ProviderCompletionInvalidError("forwarded Codex notify argv is malformed") from exc
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or not all(isinstance(value, str) for value in raw_argv)
        or not raw_argv[0]
        or any("\x00" in value for value in raw_argv)
    ):
        raise ProviderCompletionInvalidError("forwarded Codex notify argv is invalid")

    subprocess.Popen(
        [*raw_argv, notification_json],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; errors are silent to avoid leaking provider content.

    For Codex composition, capture is attempted before forwarding.  Forwarding
    is still attempted after malformed/uncorrelated capture so CAO never
    suppresses an existing notifier, while a forwarding spawn failure cannot
    undo an already atomically retained CAO report.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    failed = False
    try:
        if args.provider == "codex":
            if args.notification_json is None:
                raise ProviderCompletionInvalidError("Codex completion JSON is missing")
            if len(args.notification_json.encode("utf-8", errors="strict")) > MAX_REPORT_BYTES:
                raise ProviderCompletionInvalidError("Codex completion JSON exceeds size limit")
            notification = json.loads(args.notification_json)
            if not isinstance(notification, dict):
                raise ProviderCompletionInvalidError("Codex completion JSON is not an object")
            ingest_codex_completion(args.terminal_id, args.completion_id, notification)
        else:
            if any(
                value is None for value in (args.turn_id, args.input_message, args.final_response)
            ):
                raise ProviderCompletionInvalidError("mock completion fields are missing")
            ingest_mock_completion(
                args.terminal_id,
                args.completion_id,
                args.turn_id,
                args.input_message,
                args.final_response,
            )
    except (ProviderCompletionError, UnicodeError, json.JSONDecodeError, OSError, ValueError):
        logger.exception("Authoritative provider completion ingestion failed")
        failed = True

    if (
        args.provider == "codex"
        and args.forward_notify_json is not None
        and args.notification_json is not None
    ):
        try:
            _forward_codex_notification(args.forward_notify_json, args.notification_json)
        except (ProviderCompletionError, UnicodeError, OSError, ValueError):
            logger.exception("Forwarding the existing Codex notifier failed")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised by process-level E2E
    raise SystemExit(main())
