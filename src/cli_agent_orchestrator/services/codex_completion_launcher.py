"""Launch an assigned Codex worker with lossless notify composition.

This tiny wrapper runs inside the worker pane immediately before Codex.  That
placement is important: shell startup may choose a different ``HOME`` or
``CODEX_HOME`` from cao-server, so resolving configuration in the server process
could preserve the wrong notifier while silently replacing the actual one.

After resolving Codex's effective lower-precedence notifier, this process adds
one CLI ``notify`` override whose argv points at CAO's completion-report adapter
and embeds the prior argv for forwarding.  It then replaces itself with Codex;
there is no intermediary process after launch and it invokes no additional
shell.  The pane's normal login shell performs the one shlex-quoted CAO launch;
the launcher then uses ``execvp`` with the already separated Codex argv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Sequence

from cli_agent_orchestrator.providers.codex import (
    ProviderError,
    _resolve_existing_codex_notify,
    _toml_notify_argv,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Codex with authoritative completion capture"
    )
    parser.add_argument("--terminal-id", required=True)
    parser.add_argument("--completion-id", required=True)
    parser.add_argument("--profile-name")
    parser.add_argument("--inline-notify-json")
    parser.add_argument("codex_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.codex_argv and args.codex_argv[0] == "--":
        args.codex_argv = args.codex_argv[1:]
    if not args.codex_argv or args.codex_argv[0] != "codex":
        parser.error("the wrapped command must begin with codex")
    return args


def _inline_config(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("agent profile codexConfig notify JSON is malformed") from exc
    return {"notify": value}


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve notify in the pane environment, then ``exec`` the real Codex."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        forwarded_notify = _resolve_existing_codex_notify(
            profile_name=args.profile_name,
            inline_config=_inline_config(args.inline_notify_json),
        )
        capture_argv = [
            sys.executable,
            "-m",
            "cli_agent_orchestrator.services.provider_completion_report",
            "--provider",
            "codex",
            "--terminal-id",
            args.terminal_id,
            "--completion-id",
            args.completion_id,
        ]
        if forwarded_notify is not None:
            capture_argv.extend(
                [
                    "--forward-notify-json",
                    json.dumps(
                        forwarded_notify,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
        notify_toml = _toml_notify_argv(capture_argv, source="CAO completion adapter")
        codex_argv = [*args.codex_argv, "-c", f"notify={notify_toml}"]
        os.execvp(codex_argv[0], codex_argv)
    except (OSError, UnicodeError, ValueError, ProviderError) as exc:
        # Never print config or provider payloads.  The error identifies only
        # the failed layer/shape and keeps assigned-worker startup fail-closed.
        logger.error("Cannot launch Codex assigned worker with completion capture: %s", exc)
        return 1
    return 0  # reached only by test doubles replacing os.execvp


if __name__ == "__main__":  # pragma: no cover - exercised by live assigned workers
    raise SystemExit(main())
