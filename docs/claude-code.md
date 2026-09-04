# Claude Code Provider

## Overview

The Claude Code provider enables CLI Agent Orchestrator (CAO) to work with **Claude Code** (Anthropic's CLI) through your Anthropic API key or Claude subscription, allowing you to orchestrate multiple Claude-based agents.

## Quick Start

### Prerequisites

1. **Anthropic API Key** or **Claude Subscription**: Authentication for Claude Code
2. **Claude Code CLI**: Install the CLI tool
3. **tmux**: Required for terminal management

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Authenticate
claude setup-token
```

### Using Claude Code Provider with CAO

```bash
# Start the CAO server
cao-server

# Launch a Claude Code-backed session
cao launch --agents developer --provider claude_code
```

Via HTTP API:

```bash
curl -X POST "http://localhost:9889/sessions?provider=claude_code&agent_profile=developer"
```

## Features

### Authoritative assigned-worker completion

An `assign`-created Claude worker uses Claude Code's supported structured SDK
transport instead of the interactive TUI:

```text
claude --print --input-format stream-json --output-format stream-json --verbose \
  --session-id <CAO-bound-UUID>
```

CAO launches that command with real stdin/stdout pipes, performs the Agent SDK
`control_request: initialize` handshake, then sends each logical CAO message as
one SDK `user` JSONL record. The assignment record carries a deterministic UUID
derived from the immutable terminal ID, completion ID, and SHA-256 of the exact
post-injection task bytes. Claude echoes that UUID in `user_message_uuid` (and,
when present, `user_message_uuids`) on its terminal `ResultMessage`.

The adapter was audited and exercised against installed Claude Code 2.1.259. It
retains only fields exposed by that interface:

- `session_id`, result `uuid`, `user_message_uuid`, and optional
  `user_message_uuids` correlation identities;
- `subtype`, `is_error`, and `terminal_reason` outcome evidence; and
- the exact `result` string and SHA-256 of its strict UTF-8 bytes.

A successful callback requires `subtype=success`, `is_error=false`,
`terminal_reason=completed`, a non-empty exact result, the expected deterministic
session, and exactly one matching assigned-input identity. Error subtypes are
retained as failures; `aborted_streaming` and `aborted_tools` are retained as
cancellations; other non-success terminal boundaries are retained as
terminations. None can create a success callback.

The launch adapter atomically stores the report before publishing a small
response-free completion edge to status monitoring. That edge is only a wakeup:
the assigned-worker completion service independently reloads and validates the
retained provider report. The initialize control response is consumed rather
than written to terminal logs because it can contain account and capability
metadata.

There is deliberately no callback fallback to capture-pane, scrollback, TUI
message extraction, assignment text, prompt echo, transcript content, or a
last-line heuristic. Unknown or malformed structured fields fail closed. A
later Claude version whose ResultMessage contract changes must be audited before
the adapter's closed validation is broadened.

Ordinary operator-launched Claude terminals remain on the interactive TUI path.
Explicit `send_message` is also unchanged. If an equivalent explicit final wins
the existing completion-id transaction before the automatic report is handled,
CAO preserves intermediate explicit messages and suppresses duplicate final
callback semantics.

### Status Detection

For ordinary interactive terminals, the Claude Code provider detects terminal
states by analyzing output patterns:

- **IDLE**: Terminal shows `>` or `❯` prompt, ready for input
- **PROCESSING**: Spinner characters visible (`✶`, `✢`, `✽`, `✻`, `·`, `✳`) with ellipsis and status text
- **WAITING_USER_ANSWER**: Claude showing numbered selection options with `❯` cursor
- **COMPLETED**: Response marker `⏺` present + idle prompt visible
- **ERROR**: No recognizable output state

Status detection checks patterns in priority order: PROCESSING → WAITING_USER_ANSWER → COMPLETED → IDLE → ERROR.

### Message Extraction

For ordinary interactive uses, the provider can extract the last assistant
response by finding the `⏺` response marker. This parser is never used to create
an assigned-worker completion callback:

1. Find all `⏺` markers in the output
2. Take the last one (final response)
3. Extract text until the next `>` prompt or separator line (`────────`)
4. Strip ANSI codes from the result

### Permission Bypass

By default, CAO launches Claude Code with `--dangerously-skip-permissions` to bypass:
- **Workspace trust dialog**: The "Yes, I trust this folder" prompt that appears for new directories
- **Tool permission prompts**: Approval dialogs for file edits, command execution, etc.

This is safe because CAO already confirms workspace trust during `cao launch` ("Do you trust all the actions in this folder?") or via `--yolo` flag. Without this flag, worker agents spawned via handoff/assign would block on the trust dialog with no way to accept it interactively.

Profiles can opt into a stricter behavior by setting the `permissionMode` field, which causes the provider to pass `--permission-mode <value>` instead of `--dangerously-skip-permissions`. See [Permission Mode Override](#permission-mode-override) below. `permissionMode` takes priority over `--yolo`; when set, the provider always uses `--permission-mode <value>` regardless of yolo. When running as root/sudo, `--dangerously-skip-permissions` is omitted even in yolo mode because Claude Code rejects it under root.

A fallback `_handle_trust_prompt()` method also monitors for the trust dialog and sends Enter to accept it, in case the flag doesn't cover all scenarios.

## Configuration

### Agent Profile Integration

When launched with an agent profile (e.g., `--agents code_supervisor`), CAO:

1. Loads the profile from the agent store
2. Extracts the system prompt from the Markdown content
3. Passes it via a private `--append-system-prompt-file`
4. Injects MCP servers via `--mcp-config` JSON if the profile defines `mcpServers`

### Launch Command

The provider builds the command via `_build_claude_command()`:

```
claude --dangerously-skip-permissions [--append-system-prompt "..."] [--mcp-config "..."]
claude --permission-mode auto [--append-system-prompt "..."] [--mcp-config "..."]
```

### Permission Mode Override

The `permissionMode` field on an agent profile lets you replace the default `--dangerously-skip-permissions` bypass with a stricter Claude Code permission tier.

Allowed values: `default`, `acceptEdits`, `plan`, `auto`, `bypassPermissions`. See the [Claude Code permission modes reference](https://code.claude.com/docs/en/permission-modes) for what each tier does.

When set, the provider passes `--permission-mode <value>` instead of `--dangerously-skip-permissions`. `permissionMode` takes priority over `--yolo`; the provider always uses `--permission-mode <value>` when the field is set, even in yolo mode.

Example — a reviewer that runs under the `auto` permission classifier instead of unconditional bypass:

```markdown
---
name: reviewer
description: Code Reviewer
provider: claude_code
role: reviewer
permissionMode: auto
---

You review code for quality and correctness.
```

## Eager Inbox Delivery

Claude Code's Ink TUI buffers pasted input even while the agent is processing. CAO exploits this to deliver queued inbox messages during PROCESSING and WAITING_USER_ANSWER states, eliminating inter-turn latency. Enable with `CAO_EAGER_INBOX_DELIVERY=true`.

See [Inbox Delivery](inbox-delivery.md) for the full architecture, two-flag gate, and how to enable this for other providers.

## Native Agent Routing

When a CAO profile specifies a `native_agent` field, the provider passes `--agent <name>` directly to Claude Code's native agent store (`~/.claude/agents/`). This is a thin-wrapper mode where Claude Code handles all configuration (MCP servers, hooks, tools, model).

If no CAO profile is found for the given agent name, the provider also falls back to `--agent <name>`, assuming it exists in the native store.

```markdown
---
name: my-wrapper
description: Thin wrapper for a native Claude Code agent
provider: claude_code
native_agent: my-native-agent
---
```

## Implementation Notes

- **Prompt patterns**: `IDLE_PROMPT_PATTERN` matches both old `>` and new `❯` prompt styles, including non-breaking space (`\xa0`)
- **ANSI handling**: All pattern matching strips ANSI codes first via `ANSI_CODE_PATTERN`
- **Processing detection**: `PROCESSING_PATTERN` matches both old format (`✽ Cooking… (esc to interrupt)`) and new Claude Code 2.x format (`✽ Cooking… (6s · ↓ 174 tokens · thinking)`)
- **Trust prompt exclusion**: `TRUST_PROMPT_PATTERN` ("Yes, I trust this folder") is excluded from `WAITING_USER_ANSWER` detection to avoid false positives during initialization
- **Shell escaping**: Uses `shlex.join()` for safe command construction with multiline prompts
- **Exit command**: `/exit` via `POST /terminals/{terminal_id}/exit`

### Status Values

- `TerminalStatus.IDLE`: Ready for input
- `TerminalStatus.PROCESSING`: Working on task
- `TerminalStatus.WAITING_USER_ANSWER`: Waiting for user input
- `TerminalStatus.COMPLETED`: Task finished
- `TerminalStatus.ERROR`: Error occurred

## End-to-End Testing

The E2E test suite validates handoff, assign, and send_message flows for Claude Code.

The completion adapter also has an opt-in process-level test that launches the
real configured Claude profiles while relocating every CAO-owned file and the
SQLite database to a temporary directory:

```bash
CAO_RUN_ACTUAL_CLAUDE_E2E=1 \
CAO_ACTUAL_CLAUDE_HOME=/path/to/provider-home \
CAO_ACTUAL_AGENT_STORE=/path/to/cao/agent-store \
uv run pytest -o addopts='' -m e2e \
  test/services/test_claude_code_completion_callback_actual_e2e.py -vv
```

The test uses the exact four-line synthetic assignment embedded in the test,
asserts zero explicit worker messages, one report/callback/inbox row per worker,
exact response bytes and hash, and retained-report equality after an isolated
server restart. It must never be pointed at a production CAO home.

### Running Claude Code E2E Tests

```bash
# Start CAO server
uv run cao-server

# Run all Claude Code E2E tests
uv run pytest -m e2e test/e2e/ -v -k claude_code

# Run specific test types
uv run pytest -m e2e test/e2e/test_handoff.py -v -k claude_code
uv run pytest -m e2e test/e2e/test_assign.py -v -k claude_code
uv run pytest -m e2e test/e2e/test_send_message.py -v -k claude_code
uv run pytest -m e2e test/e2e/test_supervisor_orchestration.py -v -k ClaudeCode -o "addopts="
```

## Troubleshooting

### Common Issues

1. **Trust Dialog Blocking**:
   - Claude Code should launch with `--dangerously-skip-permissions` automatically
   - If the trust dialog still appears, check that the provider code includes the flag

2. **Processing Detection Failure**:
   - Verify Claude Code CLI version (`claude --version`)
   - Newer versions may use different spinner formats — check `PROCESSING_PATTERN`

3. **Authentication Issues**:
   ```bash
   claude setup-token
   # Or set ANTHROPIC_API_KEY environment variable
   ```

4. **Status Stuck on ERROR**:
   - Attach to tmux session and check terminal output
   - Verify Claude Code starts correctly in a regular terminal first
