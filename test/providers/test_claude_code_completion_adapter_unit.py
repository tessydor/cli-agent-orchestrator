"""Assigned Claude Code structured-completion provider regressions."""

from __future__ import annotations

import json
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli_agent_orchestrator.models.provider import ProviderType
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
from cli_agent_orchestrator.providers.manager import ProviderManager
from cli_agent_orchestrator.services.claude_completion_launcher import (
    ADAPTER_COMPLETION_MARKER,
    ADAPTER_ERROR_MARKER,
    ADAPTER_READY_MARKER,
)
from cli_agent_orchestrator.services.provider_completion_report import claude_session_id

TERMINAL_ID = "c1a0de01"
COMPLETION_ID = "1234567890abcdef1234567890abcdef"


def _assigned_provider() -> ClaudeCodeProvider:
    return ClaudeCodeProvider(
        TERMINAL_ID,
        "cao-test",
        "worker",
        completion_id=COMPLETION_ID,
    )


def _command_argv(command: str) -> list[str]:
    """Extract the command after ClaudeProvider's inherited-env scrub prefix."""
    return shlex.split(command.rsplit("; ", 1)[1])


def test_assigned_command_uses_supported_stream_json_result_interface() -> None:
    provider = _assigned_provider()

    argv = _command_argv(provider._build_claude_command(profile=None))

    assert argv[:3] == [
        sys.executable,
        "-m",
        "cli_agent_orchestrator.services.claude_completion_launcher",
    ]
    assert argv[3:7] == [
        "--terminal-id",
        TERMINAL_ID,
        "--completion-id",
        COMPLETION_ID,
    ]
    claude_index = argv.index("claude")
    claude_argv = argv[claude_index:]
    assert "--print" in claude_argv
    assert claude_argv[claude_argv.index("--input-format") + 1] == "stream-json"
    assert claude_argv[claude_argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in claude_argv
    assert claude_argv[claude_argv.index("--session-id") + 1] == claude_session_id(
        TERMINAL_ID, COMPLETION_ID
    )


def test_ordinary_claude_terminal_remains_interactive() -> None:
    provider = ClaudeCodeProvider(TERMINAL_ID, "cao-test", "operator")

    argv = _command_argv(provider._build_claude_command(profile=None))

    assert argv[0] == "claude"
    assert "claude_completion_launcher" not in " ".join(argv)
    assert "--print" not in argv
    assert "--input-format" not in argv
    assert provider.paste_enter_count == 2
    assert provider.force_bracketed_paste is True


def test_assigned_input_is_one_exact_unicode_jsonl_record() -> None:
    provider = _assigned_provider()
    task = "first line\nКирилл — Oʻzbekiston — 東京 — 🧪\nlast line"

    encoded = provider.encode_terminal_input(task, "assign")
    payload = json.loads(encoded)

    assert "\n" not in encoded
    assert payload == {
        "type": "user",
        "message": {"role": "user", "content": task},
        "parent_tool_use_id": None,
        "session_id": claude_session_id(TERMINAL_ID, COMPLETION_ID),
        "uuid": provider._completion_input_id,
    }
    assert provider.paste_enter_count == 1
    assert provider.force_bracketed_paste is False
    assert provider.paste_submit_delay == 0.0


def test_non_assignment_input_cannot_claim_assignment_identity() -> None:
    provider = _assigned_provider()
    assigned = json.loads(provider.encode_terminal_input("task", "assign"))
    explicit = json.loads(provider.encode_terminal_input("later message", "explicit"))

    assert assigned["uuid"] == provider._completion_input_id
    assert explicit["uuid"] != assigned["uuid"]
    assert explicit["session_id"] == assigned["session_id"]


def test_assignment_or_transcript_answer_is_not_a_completion_signal() -> None:
    provider = _assigned_provider()
    expected_in_prompt = "SYNTHETIC_CLAUDE_CALLBACK_SMOKE_OK"
    encoded = provider.encode_terminal_input(
        f"Return exactly:\n{expected_in_prompt}",
        "assign",
    )
    provider.mark_input_received()
    session_id = claude_session_id(TERMINAL_ID, COMPLETION_ID)
    output = "\n".join(
        (
            encoded,  # terminal echo of the assignment JSON
            json.dumps(
                {
                    "type": "assistant",
                    "session_id": session_id,
                    "message": {"content": [{"type": "text", "text": expected_in_prompt}]},
                }
            ),
        )
    )

    backend = MagicMock()
    backend.get_native_status.return_value = None
    with patch("cli_agent_orchestrator.backends.registry._backend", backend):
        assert provider.get_status(output) == TerminalStatus.PROCESSING


def test_only_exact_correlated_result_marks_structured_turn_completed() -> None:
    provider = _assigned_provider()
    provider.encode_terminal_input("task", "assign")
    provider.mark_input_received()
    session_id = claude_session_id(TERMINAL_ID, COMPLETION_ID)
    init = json.dumps({"type": "system", "subtype": "init", "session_id": session_id})
    wrong_turn = json.dumps(
        {
            "type": "result",
            "session_id": session_id,
            "user_message_uuid": "99999999-8888-4777-8666-555555555555",
        }
    )
    right_turn = json.dumps(
        {
            "type": "result",
            "session_id": session_id,
            "user_message_uuid": provider._completion_input_id,
            "user_message_uuids": [provider._completion_input_id],
        }
    )

    backend = MagicMock()
    backend.get_native_status.return_value = None
    with patch("cli_agent_orchestrator.backends.registry._backend", backend):
        assert provider.get_status(init) == TerminalStatus.PROCESSING
        assert provider.get_status("\n".join((init, wrong_turn))) == TerminalStatus.PROCESSING
        assert provider.get_status("\n".join((init, right_turn))) == TerminalStatus.COMPLETED


def test_ambiguous_or_adapter_rejected_result_is_error() -> None:
    provider = _assigned_provider()
    provider.encode_terminal_input("task", "assign")
    provider.mark_input_received()
    session_id = claude_session_id(TERMINAL_ID, COMPLETION_ID)
    ambiguous = json.dumps(
        {
            "type": "result",
            "session_id": session_id,
            "user_message_uuid": provider._completion_input_id,
            "user_message_uuids": [
                provider._completion_input_id,
                "99999999-8888-4777-8666-555555555555",
            ],
        }
    )

    backend = MagicMock()
    backend.get_native_status.return_value = None
    with patch("cli_agent_orchestrator.backends.registry._backend", backend):
        assert provider.get_status(ambiguous) == TerminalStatus.ERROR
        assert provider.get_status(ADAPTER_ERROR_MARKER) == TerminalStatus.ERROR
        assert provider.get_status(ADAPTER_COMPLETION_MARKER) == TerminalStatus.COMPLETED


def test_successful_control_handshake_is_structured_idle() -> None:
    provider = _assigned_provider()
    backend = MagicMock()
    backend.get_native_status.return_value = None
    with patch("cli_agent_orchestrator.backends.registry._backend", backend):
        assert provider.get_status(ADAPTER_READY_MARKER) == TerminalStatus.IDLE


def test_manager_passes_completion_identity_on_create_and_restart() -> None:
    manager = ProviderManager()
    created = MagicMock(spec=ClaudeCodeProvider)
    metadata = {
        "provider": ProviderType.CLAUDE_CODE.value,
        "tmux_session": "cao-test",
        "tmux_window": "worker",
        "agent_profile": "atlas_data_worker",
        "shell_command": None,
    }

    with patch(
        "cli_agent_orchestrator.providers.manager.ClaudeCodeProvider",
        return_value=created,
    ) as provider_class:
        manager.create_provider(
            ProviderType.CLAUDE_CODE.value,
            TERMINAL_ID,
            "cao-test",
            "worker",
            "atlas_data_worker",
            completion_id=COMPLETION_ID,
        )
        provider_class.assert_called_once_with(
            TERMINAL_ID,
            "cao-test",
            "worker",
            "atlas_data_worker",
            None,
            skill_prompt=None,
            model=None,
            completion_id=COMPLETION_ID,
        )

    restarted_manager = ProviderManager()
    restored = MagicMock(spec=ClaudeCodeProvider)
    with (
        patch(
            "cli_agent_orchestrator.providers.manager.get_terminal_metadata",
            return_value=metadata,
        ),
        patch(
            "cli_agent_orchestrator.providers.manager.get_assigned_worker_callback",
            return_value=SimpleNamespace(completion_id=COMPLETION_ID),
        ),
        patch(
            "cli_agent_orchestrator.providers.manager.ClaudeCodeProvider",
            return_value=restored,
        ) as provider_class,
    ):
        assert restarted_manager.get_provider(TERMINAL_ID) is restored

    assert provider_class.call_args.kwargs["completion_id"] == COMPLETION_ID
