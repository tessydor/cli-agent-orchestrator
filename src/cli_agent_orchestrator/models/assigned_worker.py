"""Durable assigned-worker completion callback models.

The worker terminal is intentionally not the identity of a completion.  Terminal
rows are operational and may be retired, while assignments and their final reports
must remain inspectable for recovery.  ``assignment_id`` and ``completion_id`` are
therefore immutable, independently generated identifiers persisted at assignment
creation time.
"""

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssignmentLifecycle(str, Enum):
    """Lifecycle of the task itself, independent of callback delivery."""

    ASSIGNED = "assigned"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNRESOLVED = "unresolved"


class CompletionDeliveryState(str, Enum):
    """Durable state machine for the successful-completion callback."""

    NOT_READY = "not_ready"
    CAPTURED = "captured"
    DELIVERING = "delivering"
    ENQUEUED = "enqueued"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED_EXPLICIT = "suppressed_explicit"
    RETRYABLE = "retryable"
    MANUAL_RECOVERY = "manual_recovery"
    TERMINAL_ERROR = "terminal_error"


class CompletionReceiverState(str, Enum):
    """Classification of the immutable callback receiver."""

    UNKNOWN = "unknown"
    ACTIVE = "active"
    RETAINED_UNREACHABLE = "retained_unreachable"
    DELETED = "deleted"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENTLY_INVALID = "permanently_invalid"


class AssignedWorkerCallback(BaseModel):
    """Read model for one durable assigned-worker callback record."""

    assignment_id: str = Field(..., description="Immutable assignment identity")
    completion_id: str = Field(..., description="Immutable completion identity")
    worker_terminal_id: str
    caller_id: str
    routing_digest: str = Field(..., description="Immutable digest of the persisted route")
    lifecycle: AssignmentLifecycle
    delivery_state: CompletionDeliveryState
    receiver_state: CompletionReceiverState
    final_result: Optional[str] = None
    final_result_sha256: Optional[str] = None
    result_reference: Optional[str] = None
    inbox_message_id: Optional[int] = None
    attempt_count: int = 0
    created_at: datetime
    dispatched_at: Optional[datetime] = None
    completion_observed_at: Optional[datetime] = None
    captured_at: Optional[datetime] = None
    first_attempt_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    enqueued_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    terminal_error_at: Optional[datetime] = None
    last_error: Optional[str] = None


class AssignedWorkerIntegrityError(ValueError):
    """Raised when durable callback state fails integrity validation."""


_EXPLICIT_SENDER_SUFFIX_RE = re.compile(
    r"\n\n\[Message from terminal [a-f0-9]{8}\. "
    r"Use send_message MCP tool for any follow-up work\.\]\s*$"
)


def callback_routing_digest(
    assignment_id: str,
    completion_id: str,
    worker_terminal_id: str,
    caller_id: str,
) -> str:
    """Return the deterministic digest anchoring immutable callback routing.

    A canonical JSON object avoids delimiter ambiguity.  SQLite update triggers
    prevent any of these values (including the digest) from changing after the
    assignment row is created; read validation independently recomputes it.
    """
    payload = json.dumps(
        {
            "assignment_id": assignment_id,
            "caller_id": caller_id,
            "completion_id": completion_id,
            "worker_terminal_id": worker_terminal_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_callback_text(value: str) -> str:
    """Canonicalize only CAO's exact explicit-message transport suffix."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _EXPLICIT_SENDER_SUFFIX_RE.sub("", value)
    return "\n".join(line.rstrip() for line in value.strip().split("\n"))


def format_server_completion_message(
    final_result: str,
    worker_terminal_id: str,
    assignment_id: str,
    completion_id: str,
) -> str:
    """Build the stable server callback payload used for evidence validation."""
    return (
        f"{final_result}\n\n"
        "[Server-generated assigned-worker completion callback: "
        f"worker={worker_terminal_id}; assignment={assignment_id}; "
        f"completion={completion_id}]"
    )
