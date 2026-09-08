import json
import shlex
from unittest.mock import MagicMock

import pytest

from cli_agent_orchestrator.models.provider_completion import ProviderCompletionCorrelationError
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
from cli_agent_orchestrator.services import claude_native_completion as native
from cli_agent_orchestrator.services import provider_completion_report as reports

T = "abcdef01"
C = "1" * 32
P = "12345678-1234-4234-8234-123456789012"


@pytest.fixture(autouse=True)
def private_root(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "PROVIDER_COMPLETION_REPORT_DIR", tmp_path)


def event(kind, **kwargs):
    return dict(
        session_id=reports.claude_session_id(T, C), prompt_id=P, hook_event_name=kind, **kwargs
    )


def test_native_launch_and_followup_preserve_identity():
    native.configure_native(T, C)
    p = ClaudeCodeProvider(T, "test", "worker", completion_id=C)
    argv = shlex.split(p._build_claude_command(None).rsplit("; ", 1)[1])
    assert argv[0] == "claude"
    assert "--print" not in argv
    assert "--session-id" in argv
    assert set(json.loads(argv[argv.index("--settings") + 1])["hooks"]) == {
        "UserPromptSubmit",
        "Stop",
        "StopFailure",
    }
    text = 'long\nЮникод "' + ("x" * 32768)
    assert p.encode_terminal_input(text, "assign") == text
    assert p.encode_terminal_input(text, "explicit") == text
    assert p.paste_enter_count == 1 and p.force_bracketed_paste and p.supports_screen_detection
    assert ClaudeCodeProvider(T, "test", "worker", completion_id=C)._native_completion


def test_legacy_assignment_keeps_streaming():
    p = ClaudeCodeProvider(T, "test", "worker", completion_id=C)
    assert p._stream_completion
    assert json.loads(p.encode_terminal_input("task", "assign"))["message"]["content"] == "task"


@pytest.mark.parametrize("answer", ["Completed the work.", "I refuse this task."])
def test_stop_captures_actual_answer_even_without_send_message(answer):
    reports.bind_completion_dispatch("claude_code", T, C, "task")
    native.ingest(T, C, event("UserPromptSubmit", prompt="task"))
    native.ingest(T, C, event("Stop", last_assistant_message=answer))
    native.ingest(T, C, event("Stop", last_assistant_message=answer))
    assert reports.load_completion_report("claude_code", T, C).final_response == answer


def test_wrong_prompt_and_later_turn_cannot_claim_assignment():
    reports.bind_completion_dispatch("claude_code", T, C, "task")
    native.ingest(T, C, event("UserPromptSubmit", prompt="task"))
    native.ingest(T, C, event("UserPromptSubmit", prompt="different"))
    native.ingest(T, C, event("Stop", last_assistant_message="wrong"))
    with pytest.raises(reports.ProviderCompletionUnavailableError):
        reports.load_completion_report("claude_code", T, C)


def test_mismatched_session_rejected():
    with pytest.raises(ProviderCompletionCorrelationError):
        native.ingest(T, C, dict(event("Stop"), session_id="bad"))


def test_failure_retained_as_failure():
    reports.bind_completion_dispatch("claude_code", T, C, "task")
    native.ingest(T, C, event("UserPromptSubmit", prompt="task"))
    native.ingest(T, C, event("StopFailure", last_assistant_message="API failed"))
    assert reports.load_completion_report("claude_code", T, C).completion_state == "failure"


def test_shell_fallback_is_error(monkeypatch):
    native.configure_native(T, C)
    p = ClaudeCodeProvider(T, "test", "worker", completion_id=C)
    p._initialized = True
    backend = MagicMock()
    backend.get_pane_current_command.return_value = "bash"
    monkeypatch.setattr("cli_agent_orchestrator.providers.claude_code.get_backend", lambda: backend)
    assert p.get_status("old output") == TerminalStatus.ERROR


def test_failure_without_rendered_message_is_retained():
    reports.bind_completion_dispatch("claude_code", T, C, "task")
    native.ingest(T, C, event("UserPromptSubmit", prompt="task"))
    native.ingest(T, C, event("StopFailure", error="rate_limit"))
    report = reports.load_completion_report("claude_code", T, C)
    assert report.completion_state == "failure"
    assert "rate_limit" in report.final_response


def test_restored_native_worker_detects_shell_without_kiro_baseline(monkeypatch):
    from cli_agent_orchestrator.providers import manager

    native.configure_native(T, C)
    monkeypatch.setattr(
        manager,
        "get_terminal_metadata",
        lambda _: {
            "provider": "claude_code",
            "tmux_session": "test",
            "tmux_window": "worker",
            "agent_profile": None,
        },
    )
    callback = MagicMock()
    callback.completion_id = C
    monkeypatch.setattr(manager, "get_assigned_worker_callback", lambda _: callback)
    provider = manager.ProviderManager().get_provider(T)
    backend = MagicMock()
    backend.get_pane_current_command.return_value = "bash"
    monkeypatch.setattr("cli_agent_orchestrator.providers.claude_code.get_backend", lambda: backend)
    assert provider.get_status_from_screen(["old completed output"]) == TerminalStatus.ERROR


def test_new_native_assignment_rejects_resume_identity_collision():
    native.configure_native(T, C)
    with pytest.raises(ValueError, match="cannot resume"):
        ClaudeCodeProvider(T, "test", "worker", completion_id=C, resume_session_id=P)
