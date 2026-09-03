"""Durable ingestion and retrieval of authoritative provider completion data.

The module is also the process entry point used by provider-native completion
hooks.  Codex appends one structured JSON argument to the configured command at
its successful ``agent-turn-complete`` boundary.  That subprocess validates and
atomically retains the report; callback delivery later reads it through the
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
import sys
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
    return {
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
    final_response = report_payload["final_response"]
    if not final_response.strip():
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
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retain an authoritative provider completion")
    parser.add_argument("--provider", choices=("codex", "mock_cli"), required=True)
    parser.add_argument("--terminal-id", required=True)
    parser.add_argument("--completion-id", required=True)
    parser.add_argument("--turn-id")
    parser.add_argument("--input-message")
    parser.add_argument("--final-response")
    parser.add_argument("notification_json", nargs="?")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; errors are silent to avoid leaking provider content."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
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
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by process-level E2E
    raise SystemExit(main())
