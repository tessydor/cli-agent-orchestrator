"""Durable assigned-worker completion callback models.

The worker terminal is intentionally not the identity of a completion.  Terminal
rows are operational and may be retired, while assignments and their final reports
must remain inspectable for recovery.  ``assignment_id`` and ``completion_id`` are
therefore immutable, independently generated identifiers persisted at assignment
creation time.
"""

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


class CompletionDeliveryState(str, Enum):
    """Durable state machine for the successful-completion callback."""

    NOT_READY = "not_ready"
    CAPTURED = "captured"
    DELIVERING = "delivering"
    ENQUEUED = "enqueued"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED_EXPLICIT = "suppressed_explicit"
    RETRYABLE = "retryable"
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
