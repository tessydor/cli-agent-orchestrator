"""Authoritative Claude Code ResultMessage adapter regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cli_agent_orchestrator.models.provider_completion import (
    ProviderCompletionConflictError,
    ProviderCompletionCorrelationError,
    ProviderCompletionInvalidError,
    ProviderCompletionUnavailableError,
)
from cli_agent_orchestrator.services import provider_completion_report as reports
from cli_agent_orchestrator.utils import atomic_file

TERMINAL_ID = "c1a0de01"
COMPLETION_ID = "1234567890abcdef1234567890abcdef"
TASK = "Inspect the assigned Atlas dataset and report the exact outcome."


@pytest.fixture(autouse=True)
def isolated_report_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "provider-reports"
    monkeypatch.setattr(reports, "PROVIDER_COMPLETION_REPORT_DIR", root)
    monkeypatch.setattr(atomic_file, "LOCK_DIR", tmp_path / "locks")
    return root


def _bind(task: str = TASK, *, terminal_id: str = TERMINAL_ID, completion_id: str = COMPLETION_ID):
    digest = reports.bind_completion_dispatch("claude_code", terminal_id, completion_id, task)
    return digest, reports.claude_input_id(terminal_id, completion_id, digest)


def _result(
    result: object = "CLAUDE_CALLBACK_OK",
    *,
    terminal_id: str = TERMINAL_ID,
    completion_id: str = COMPLETION_ID,
    input_id: object | None = None,
    session_id: object | None = None,
    result_id: object = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
    subtype: object = "success",
    is_error: object = False,
    terminal_reason: object = "completed",
    user_message_uuids: object | None = None,
) -> dict[str, object]:
    if input_id is None:
        _, input_id = _bind(terminal_id=terminal_id, completion_id=completion_id)
    if session_id is None:
        session_id = reports.claude_session_id(terminal_id, completion_id)
    payload: dict[str, object] = {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": result,
        "terminal_reason": terminal_reason,
        "uuid": result_id,
        "session_id": session_id,
        "user_message_uuid": input_id,
        "user_message_uuids": [input_id] if user_message_uuids is None else user_message_uuids,
        # Supported fields not needed by the adapter remain deliberately
        # ignored rather than being copied into CAO's retained report.
        "duration_ms": 10,
        "duration_api_ms": 8,
        "num_turns": 1,
        "stop_reason": None,
        "total_cost_usd": 0.01,
        "usage": {},
        "modelUsage": {},
        "permission_denials": [],
    }
    return payload


@pytest.mark.parametrize(
    "style",
    ("atlas_data_worker", "atlas_equities_worker"),
)
def test_atlas_worker_style_completion_requires_no_explicit_message(style: str) -> None:
    response = f"{style}: authoritative completion"
    payload = _result(response)

    report = reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)

    assert report is not None
    assert report.final_response == response
    assert report.completion_state == "success"
    assert reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID) == report


@pytest.mark.parametrize(
    "response",
    (
        "first line\nsecond line\n",
        "Кирилл: натижа — Oʻzbekiston — 東京 — 🧪\naniq yakun",
    ),
)
def test_exact_multiline_and_unicode_result_bytes_are_preserved(response: str) -> None:
    report = reports.ingest_claude_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _result(response),
    )

    assert report is not None
    assert report.final_response == response
    assert report.final_response_sha256 == hashlib.sha256(response.encode("utf-8")).hexdigest()


def test_assignment_and_transcript_expected_answer_never_replace_result() -> None:
    expected_in_prompt = "SYNTHETIC_CLAUDE_CALLBACK_SMOKE_OK"
    task = f"The assignment contains {expected_in_prompt}, but return the actual finding."
    _, input_id = _bind(task)
    payload = _result("THE_REAL_FINAL_RESULT", input_id=input_id)
    payload["transcript"] = [
        {"role": "assistant", "content": expected_in_prompt},
        {"role": "user", "content": task},
    ]

    report = reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)

    assert report is not None
    assert report.final_response == "THE_REAL_FINAL_RESULT"
    assert expected_in_prompt not in report.final_response
    assert task not in report.final_response


@pytest.mark.parametrize("empty", (None, "", " \n\t"))
def test_empty_success_result_never_creates_report(empty: object) -> None:
    with pytest.raises(ProviderCompletionInvalidError):
        reports.ingest_claude_completion(
            TERMINAL_ID,
            COMPLETION_ID,
            _result(empty),
        )
    with pytest.raises(ProviderCompletionUnavailableError):
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)


@pytest.mark.parametrize(
    ("payload_updates", "expected_state"),
    (
        (
            {
                "subtype": "error_during_execution",
                "is_error": True,
                "terminal_reason": "model_error",
                "result": "",
            },
            "failure",
        ),
        (
            {
                "subtype": "success",
                "is_error": False,
                "terminal_reason": "aborted_streaming",
                "result": "partial text",
            },
            "cancelled",
        ),
        (
            {
                "subtype": "success",
                "is_error": False,
                "terminal_reason": "max_turns",
                "result": "partial text",
            },
            "terminated",
        ),
    ),
)
def test_non_success_outcomes_are_retained_but_never_marked_success(
    payload_updates: dict[str, object], expected_state: str
) -> None:
    payload = _result()
    payload.update(payload_updates)

    report = reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)

    assert report is not None
    assert report.completion_state == expected_state
    assert (
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID).completion_state
        == expected_state
    )


def test_wrong_session_id_is_rejected() -> None:
    with pytest.raises(ProviderCompletionCorrelationError, match="session"):
        reports.ingest_claude_completion(
            TERMINAL_ID,
            COMPLETION_ID,
            _result(session_id="99999999-8888-4777-8666-555555555555"),
        )


def test_wrong_run_input_identity_is_ignored_and_cannot_satisfy_assignment() -> None:
    _bind()
    payload = _result(input_id="99999999-8888-4777-8666-555555555555")

    assert reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload) is None
    with pytest.raises(ProviderCompletionUnavailableError):
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)


def test_batched_input_identity_is_ambiguous_and_rejected() -> None:
    _, input_id = _bind()
    payload = _result(
        input_id=input_id,
        user_message_uuids=[input_id, "99999999-8888-4777-8666-555555555555"],
    )

    with pytest.raises(ProviderCompletionCorrelationError, match="combined"):
        reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)


def test_duplicate_result_is_idempotent_and_different_result_identity_conflicts() -> None:
    payload = _result()
    first = reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)
    duplicate = reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)
    assert duplicate == first

    conflicting = dict(payload)
    conflicting["uuid"] = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    with pytest.raises(ProviderCompletionConflictError):
        reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, conflicting)
    with pytest.raises(ProviderCompletionConflictError):
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)


@pytest.mark.parametrize(
    "field,value",
    (
        ("type", "assistant"),
        ("subtype", "future_unknown_subtype"),
        ("is_error", "false"),
        ("uuid", "contains spaces"),
        ("terminal_reason", []),
    ),
)
def test_malformed_result_message_fails_closed(field: str, value: object) -> None:
    payload = _result()
    payload[field] = value
    with pytest.raises(ProviderCompletionInvalidError):
        reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, payload)


def test_retained_claude_report_recovers_statelessly_after_restart() -> None:
    original = reports.ingest_claude_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _result("retained exact result"),
    )

    recovered = reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)

    assert recovered == original
    assert recovered is not None
    assert recovered.final_response == "retained exact result"
    assert recovered.provider_input_id is not None
    assert recovered.provider_turn_id == "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def test_retained_state_cannot_contradict_result_message_evidence(
    isolated_report_root: Path,
) -> None:
    reports.ingest_claude_completion(
        TERMINAL_ID,
        COMPLETION_ID,
        _result(
            "failure detail",
            subtype="error_during_execution",
            is_error=True,
            terminal_reason="model_error",
        ),
    )
    report_path = (
        isolated_report_root / "claude_code" / TERMINAL_ID / f"{COMPLETION_ID}.report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["completion_state"] = "success"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderCompletionInvalidError, match="contradicts"):
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)


@pytest.mark.parametrize(
    "field",
    (
        "completion_state",
        "provider_input_id",
        "provider_result_subtype",
        "provider_terminal_reason",
        "provider_is_error",
    ),
)
def test_retained_claude_report_requires_all_outcome_and_correlation_fields(
    isolated_report_root: Path,
    field: str,
) -> None:
    reports.ingest_claude_completion(TERMINAL_ID, COMPLETION_ID, _result())
    report_path = (
        isolated_report_root / "claude_code" / TERMINAL_ID / f"{COMPLETION_ID}.report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.pop(field)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderCompletionInvalidError, match="missing"):
        reports.load_completion_report("claude_code", TERMINAL_ID, COMPLETION_ID)
