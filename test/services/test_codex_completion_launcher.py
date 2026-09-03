"""Tests for lossless Codex notify composition at assigned-worker launch."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from cli_agent_orchestrator.providers import codex as codex_provider
from cli_agent_orchestrator.services import codex_completion_launcher as launcher


@pytest.fixture(autouse=True)
def isolated_codex_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    codex_home = tmp_path / "pane-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(codex_provider, "CODEX_SYSTEM_CONFIG_PATH", tmp_path / "system-config.toml")
    monkeypatch.setattr(
        codex_provider, "CODEX_MANAGED_CONFIG_PATH", tmp_path / "managed-config.toml"
    )
    return codex_home


def _launch_args(*extra: str) -> list[str]:
    return [
        "--terminal-id",
        "deadbeef",
        "--completion-id",
        "a" * 32,
        *extra,
        "--",
        "codex",
        "--yolo",
        "--no-alt-screen",
    ]


@patch("cli_agent_orchestrator.services.codex_completion_launcher.os.execvp")
def test_launcher_composes_effective_notify_and_execs_codex_directly(
    mock_execvp, isolated_codex_config: Path
) -> None:
    original_notify = [
        "/opt/notifier with spaces/bin/notify",
        "--literal",
        "semicolon; dollar$(not-a-shell)",
    ]
    (isolated_codex_config / "config.toml").write_text(
        'notify = ["/opt/notifier with spaces/bin/notify", "--literal", '
        '"semicolon; dollar$(not-a-shell)"]\n',
        encoding="utf-8",
    )

    assert launcher.main(_launch_args()) == 0

    executable, codex_argv = mock_execvp.call_args.args
    assert executable == "codex"
    assert codex_argv[:3] == ["codex", "--yolo", "--no-alt-screen"]
    assert codex_argv[-2] == "-c"
    capture_argv = tomllib.loads(codex_argv[-1])["notify"]
    assert capture_argv[:3] == [
        launcher.sys.executable,
        "-m",
        "cli_agent_orchestrator.services.provider_completion_report",
    ]
    assert capture_argv[capture_argv.index("--terminal-id") + 1] == "deadbeef"
    assert capture_argv[capture_argv.index("--completion-id") + 1] == "a" * 32
    forward_index = capture_argv.index("--forward-notify-json")
    assert json.loads(capture_argv[forward_index + 1]) == original_notify


@patch("cli_agent_orchestrator.services.codex_completion_launcher.os.execvp")
def test_launcher_inline_notify_wins_profile_and_is_preserved_as_argv(
    mock_execvp, isolated_codex_config: Path
) -> None:
    (isolated_codex_config / "reviewer.config.toml").write_text(
        'notify = ["profile-notifier"]\n', encoding="utf-8"
    )
    inline_notify = ["inline notifier", "", "*.txt", "$(literal)"]

    assert (
        launcher.main(
            _launch_args(
                "--profile-name",
                "reviewer",
                "--inline-notify-json",
                json.dumps(inline_notify),
            )
        )
        == 0
    )

    codex_argv = mock_execvp.call_args.args[1]
    capture_argv = tomllib.loads(codex_argv[-1])["notify"]
    forward_index = capture_argv.index("--forward-notify-json")
    assert json.loads(capture_argv[forward_index + 1]) == inline_notify


@patch("cli_agent_orchestrator.services.codex_completion_launcher.os.execvp")
def test_launcher_rejects_managed_notify_without_starting_codex(
    mock_execvp, tmp_path: Path
) -> None:
    codex_provider.CODEX_MANAGED_CONFIG_PATH.write_text(
        'notify = ["managed-notifier"]\n', encoding="utf-8"
    )

    assert launcher.main(_launch_args()) == 1
    mock_execvp.assert_not_called()


@patch("cli_agent_orchestrator.services.codex_completion_launcher.os.execvp")
def test_launcher_rejects_malformed_inline_notify_without_starting_codex(mock_execvp) -> None:
    assert launcher.main(_launch_args("--inline-notify-json", '"not-an-array"')) == 1
    mock_execvp.assert_not_called()
