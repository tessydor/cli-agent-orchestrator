"""Provider-neutral authoritative completion-report contract.

Terminal text is intentionally absent from this contract.  A provider adapter
must return a structured report emitted by the provider at its native successful
turn boundary, correlated to the exact CAO dispatch.  Providers without such a
source inherit the fail-closed ``BaseProvider`` implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


class ProviderCompletionError(RuntimeError):
    """Base class for failures at the authoritative completion boundary."""


class ProviderCompletionUnavailableError(ProviderCompletionError):
    """No authoritative report exists yet; retry may recover it."""


class ProviderCompletionInvalidError(ProviderCompletionError):
    """A report exists but violates the provider-neutral contract."""


class ProviderCompletionCorrelationError(ProviderCompletionInvalidError):
    """A report does not belong to the exact dispatched worker turn."""


class ProviderCompletionConflictError(ProviderCompletionInvalidError):
    """Two non-identical reports claimed the same immutable completion."""


class ProviderCompletionTerminalOutcomeError(ProviderCompletionError):
    """A correlated provider turn ended without a successful completion.

    This is deliberately distinct from an invalid report.  A failed,
    cancelled, or otherwise terminated provider ResultMessage is authoritative
    outcome evidence; it must not create a callback, but it also must not be
    misclassified as malformed evidence requiring manual interpretation.
    """

    def __init__(
        self,
        completion_state: str,
        provider_result_subtype: str | None,
        provider_terminal_reason: str | None,
    ) -> None:
        self.completion_state = completion_state
        self.provider_result_subtype = provider_result_subtype
        self.provider_terminal_reason = provider_terminal_reason
        super().__init__(
            "provider completion ended with "
            f"state={completion_state}, subtype={provider_result_subtype}, "
            f"terminal_reason={provider_terminal_reason}"
        )


def utf8_sha256(value: str) -> str:
    """Hash the exact strict UTF-8 bytes of ``value`` without normalization."""
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def input_messages_sha256(input_messages: Sequence[str]) -> str:
    """Hash an ordered provider input-message vector canonically.

    The vector digest is provenance only.  Exact-turn correlation uses the
    digest of the final structured input message, which is the newly dispatched
    turn in Codex's cumulative ``input-messages`` payload.
    """
    encoded = json.dumps(
        list(input_messages),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProviderCompletionReport:
    """One immutable, authoritative provider completion report."""

    provider: str
    terminal_id: str
    completion_id: str
    provider_session_id: str
    provider_turn_id: str
    input_messages_sha256: str
    dispatched_input_sha256: str
    final_response: str
    final_response_sha256: str
    source_reference: str
    # Additive provider-neutral outcome/correlation fields.  Legacy Codex V1
    # reports omit them on disk and load with the success defaults below, which
    # preserves retained-report recovery across this maintenance update.
    completion_state: str = "success"
    provider_input_id: str | None = None
    provider_result_subtype: str | None = None
    provider_terminal_reason: str | None = None
    provider_is_error: bool | None = None
