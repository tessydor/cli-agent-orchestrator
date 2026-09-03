"""Authoritative provider completion-report contract regressions."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.provider_completion import (
    ProviderCompletionConflictError,
    ProviderCompletionCorrelationError,
    ProviderCompletionInvalidError,
    ProviderCompletionUnavailableError,
)
from cli_agent_orchestrator.services import provider_completion_report as reports
from cli_agent_orchestrator.utils import atomic_file

TERMINAL_ID = "a1b2c3d4"
COMPLETION_ID = "0123456789abcdef0123456789abcdef"
TASK = "Reply with exactly SYNTHETIC_CALLBACK_SMOKE_OK"


@pytest.fixture(autouse=True)
def isolated_report_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "provider-reports"
    monkeypatch.setattr(reports, "PROVIDER_COMPLETION_REPORT_DIR", root)
    monkeypatch.setattr(atomic_file, "LOCK_DIR", tmp_path / "locks")
    return root


def _codex_payload(
    response: object = "SYNTHETIC_CALLBACK_SMOKE_OK",
    *,
    input_messages: object = None,
    thread_id: object = "11111111-2222-3333-4444-555555555555",
    turn_id: object = "turn-1",
) -> dict[str, object]:
    return {
        "type": "agent-turn-complete",
        "thread-id": thread_id,
        "turn-id": turn_id,
        "cwd": "/isolated/test",
        "input-messages": [TASK] if input_messages is None else input_messages,
        "last-assistant-message": response,
    }


def _bind(task: str = TASK) -> str:
    return reports.bind_completion_dispatch("codex", TERMINAL_ID, COMPLETION_ID, task)


def test_exact_synthetic_response_and_utf8_digest_are_authoritative() -> None:
    _bind()
    reports.ingest_codex_completion(TERMINAL_ID, COMPLETION_ID, _codex_payload())

    report = reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)

    assert report.final_response == "SYNTHETIC_CALLBACK_SMOKE_OK"
    assert (
        report.final_response_sha256 == hashlib.sha256(b"SYNTHETIC_CALLBACK_SMOKE_OK").hexdigest()
    )
    assert report.provider_session_id == "11111111-2222-3333-4444-555555555555"
    assert report.provider_turn_id == "turn-1"


@pytest.mark.parametrize(
    "response",
    (
        "first line\nsecond line\n",
        "Unicode: café — 東京 — 🧪\nexact trailing line",
    ),
)
def test_multiline_and_unicode_response_bytes_are_not_normalized(response: str) -> None:
    _bind()
    reports.ingest_codex_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _codex_payload(response),
    )

    report = reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)
    assert report.final_response == response
    assert report.final_response_sha256 == hashlib.sha256(response.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("response", (None, "", " \n\t"))
def test_empty_or_missing_final_response_never_creates_a_success_report(response: object) -> None:
    _bind()

    with pytest.raises(ProviderCompletionInvalidError):
        reports.ingest_codex_completion(
            TERMINAL_ID,
            COMPLETION_ID,
            _codex_payload(response),
        )

    with pytest.raises(ProviderCompletionUnavailableError):
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"type": "not-a-completion"},
        _codex_payload(input_messages=[]),
        _codex_payload(input_messages="not-an-array"),
        _codex_payload(thread_id=None),
        _codex_payload(turn_id="contains spaces"),
    ),
)
def test_malformed_or_missing_codex_completion_fails_closed(payload: dict[str, object]) -> None:
    _bind()
    with pytest.raises(ProviderCompletionInvalidError):
        reports.ingest_codex_completion(TERMINAL_ID, COMPLETION_ID, payload)


def test_wrong_dispatch_correlation_is_rejected() -> None:
    _bind("the exact dispatched input")
    reports.ingest_codex_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _codex_payload("real response", input_messages=["a different turn"]),
    )

    with pytest.raises(ProviderCompletionCorrelationError):
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)


def test_prior_transcript_and_prompt_token_never_substitute_for_final_response() -> None:
    task = "Do the task and end with SYNTHETIC_CALLBACK_SMOKE_OK"
    _bind(task)
    reports.ingest_codex_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _codex_payload(
            "THE_REAL_FINAL_ASSISTANT_RESPONSE",
            input_messages=[
                "Earlier transcript said SYNTHETIC_CALLBACK_SMOKE_OK",
                task,
            ],
        ),
    )

    report = reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)
    assert report.final_response == "THE_REAL_FINAL_ASSISTANT_RESPONSE"
    assert "SYNTHETIC_CALLBACK_SMOKE_OK" not in report.final_response


def test_duplicate_report_is_idempotent_and_conflicting_duplicate_is_retained() -> None:
    _bind()
    payload = _codex_payload()
    reports.ingest_codex_completion(TERMINAL_ID, COMPLETION_ID, payload)
    reports.ingest_codex_completion(TERMINAL_ID, COMPLETION_ID, payload)
    assert reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID).final_response == (
        "SYNTHETIC_CALLBACK_SMOKE_OK"
    )

    with pytest.raises(ProviderCompletionConflictError):
        reports.ingest_codex_completion(
            TERMINAL_ID,
            COMPLETION_ID,
            _codex_payload("different response", turn_id="turn-2"),
        )
    with pytest.raises(ProviderCompletionConflictError):
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)


def test_retained_report_loads_after_new_adapter_instance_and_files_are_private(
    isolated_report_root: Path,
) -> None:
    _bind()
    reports.ingest_codex_completion(TERMINAL_ID, COMPLETION_ID, _codex_payload())

    # Retrieval is stateless and filesystem-backed, which is the restart and
    # retained-report recovery contract used by an on-demand provider adapter.
    recovered = reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)
    assert recovered.final_response == "SYNTHETIC_CALLBACK_SMOKE_OK"

    files = sorted(isolated_report_root.rglob("*.json"))
    assert len(files) == 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)
    directories = [isolated_report_root, *isolated_report_root.rglob("*")]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories if path.is_dir())


def test_malformed_retained_report_is_not_read_as_text() -> None:
    _bind()
    _, report_path, _ = reports._paths("codex", TERMINAL_ID, COMPLETION_ID)
    report_path.write_bytes(b"{not valid json")

    with pytest.raises(ProviderCompletionInvalidError, match="malformed"):
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)


def test_codex_hook_cli_rejects_missing_or_malformed_payload_without_writing() -> None:
    common = [
        "--provider",
        "codex",
        "--terminal-id",
        TERMINAL_ID,
        "--completion-id",
        COMPLETION_ID,
    ]
    assert reports.main(common) == 1
    assert reports.main([*common, "not-json"]) == 1
    assert reports.main([*common, json.dumps(_codex_payload())]) == 0


@patch("cli_agent_orchestrator.services.provider_completion_report.subprocess.Popen")
def test_composed_notify_forwards_identical_json_as_direct_argv(mock_popen) -> None:
    _bind()
    notification_json = json.dumps(_codex_payload(), ensure_ascii=False)
    forward_argv = [
        "/opt/notifier with spaces/bin/notify",
        "--literal",
        "semicolon; dollar$(not-a-shell)",
    ]
    result = reports.main(
        [
            "--provider",
            "codex",
            "--terminal-id",
            TERMINAL_ID,
            "--completion-id",
            COMPLETION_ID,
            "--forward-notify-json",
            json.dumps(forward_argv),
            notification_json,
        ]
    )

    assert result == 0
    mock_popen.assert_called_once_with(
        [*forward_argv, notification_json],
        stdin=reports.subprocess.DEVNULL,
        stdout=reports.subprocess.DEVNULL,
        stderr=reports.subprocess.DEVNULL,
        shell=False,
    )
    mock_popen.return_value.wait.assert_not_called()
    mock_popen.return_value.poll.assert_not_called()
    assert (
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID).final_response
        == "SYNTHETIC_CALLBACK_SMOKE_OK"
    )


@patch(
    "cli_agent_orchestrator.services.provider_completion_report.subprocess.Popen",
    side_effect=OSError("notifier spawn failed"),
)
def test_forward_spawn_failure_cannot_undo_cao_capture(mock_popen) -> None:
    _bind()
    notification_json = json.dumps(_codex_payload())

    result = reports.main(
        [
            "--provider",
            "codex",
            "--terminal-id",
            TERMINAL_ID,
            "--completion-id",
            COMPLETION_ID,
            "--forward-notify-json",
            json.dumps(["missing-notifier"]),
            notification_json,
        ]
    )

    assert result == 1
    mock_popen.assert_called_once()
    report = reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)
    assert report.final_response == "SYNTHETIC_CALLBACK_SMOKE_OK"


@patch("cli_agent_orchestrator.services.provider_completion_report.subprocess.Popen")
def test_capture_failure_does_not_suppress_existing_notifier(mock_popen) -> None:
    _bind()
    malformed_completion = json.dumps(_codex_payload(response=""))
    forward_argv = ["existing-notifier", "--still-runs"]

    result = reports.main(
        [
            "--provider",
            "codex",
            "--terminal-id",
            TERMINAL_ID,
            "--completion-id",
            COMPLETION_ID,
            "--forward-notify-json",
            json.dumps(forward_argv),
            malformed_completion,
        ]
    )

    assert result == 1
    mock_popen.assert_called_once_with(
        [*forward_argv, malformed_completion],
        stdin=reports.subprocess.DEVNULL,
        stdout=reports.subprocess.DEVNULL,
        stderr=reports.subprocess.DEVNULL,
        shell=False,
    )
    with pytest.raises(ProviderCompletionUnavailableError):
        reports.load_completion_report("codex", TERMINAL_ID, COMPLETION_ID)
