"""Regression: submitting a Claude prompt must not also answer its next menu."""

from unittest.mock import patch

from cli_agent_orchestrator.clients.tmux import TmuxClient
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider


def test_prompt_submission_leaves_next_menu_unanswered():
    provider = ClaudeCodeProvider("c1a0de01", "cao-test", "worker")
    state = {"screen": "editor", "answers": [], "payload": None}

    def run(argv, **kwargs):
        if argv[:2] == ["tmux", "load-buffer"]:
            state["payload"] = kwargs["input"]
        elif argv[:2] == ["tmux", "send-keys"] and argv[-1] == "Enter":
            if state["screen"] == "editor":
                state["screen"] = "menu"
            else:
                state["answers"].append("default")

    with (
        patch("cli_agent_orchestrator.clients.tmux.libtmux"),
        patch("cli_agent_orchestrator.clients.tmux.subprocess.run", side_effect=run),
        patch("cli_agent_orchestrator.clients.tmux.time.sleep") as sleep,
        patch.object(TmuxClient, "_paste_buffer_sanitizes", True),
    ):
        client = TmuxClient()
        client.send_keys(
            "cao-test",
            "worker",
            "Inspect only\n東京 — Тест",
            enter_count=provider.paste_enter_count,
            force_bracketed_paste=provider.force_bracketed_paste,
            submit_delay=provider.paste_submit_delay,
        )

    assert state["payload"] == "Inspect only\n東京 — Тест".encode()
    assert state["screen"] == "menu"
    assert state["answers"] == [], "Prompt submission also answered the next menu"
    sleep.assert_called_once_with(2.0)
