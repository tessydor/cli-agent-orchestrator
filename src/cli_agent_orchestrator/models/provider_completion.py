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
