"""Inbox message models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrchestrationType(str, Enum):
    """Orchestration mode for a message delivery."""

    SEND_MESSAGE = "send_message"
    HANDOFF = "handoff"
    ASSIGN = "assign"


class MessageStatus(str, Enum):
    """Message status enumeration."""

    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    # correction-1038: a PENDING row explicitly, durably superseded by a
    # later PENDING row from the SAME (sender_id, receiver_id) pair, via
    # ``clients.database.supersede_inbox_messages``. The original row/
    # content/id/sender/receiver are never altered or deleted -- only this
    # status and ``superseded_by_message_id`` change. Excluded from
    # ``get_pending_messages``/delivery exactly because it is no longer
    # PENDING; never claimed, never DELIVERED, never fabricated.
    SUPERSEDED = "superseded"


class InboxMessageOrigin(str, Enum):
    """Source of an inbox row, retained for callback deduplication/audit."""

    EXPLICIT = "explicit"
    SERVER_COMPLETION = "server_completion"
    SYSTEM = "system"
    LEGACY = "legacy"


class InboxMessage(BaseModel):
    """Inbox message model."""

    id: int = Field(..., description="Message ID")
    sender_id: str = Field(..., description="Sender terminal ID")
    receiver_id: str = Field(..., description="Receiver terminal ID")
    message: str = Field(..., description="Message content")
    status: MessageStatus = Field(..., description="Message status")
    created_at: datetime = Field(..., description="Creation timestamp")
    origin: InboxMessageOrigin = Field(
        InboxMessageOrigin.LEGACY, description="Producer of this durable inbox row"
    )
    assignment_id: str | None = Field(
        None, description="Assigned-worker identity when this message belongs to one"
    )
    idempotency_key: str | None = Field(
        None, description="Server-side deduplication key for idempotent producers"
    )
    claim_token: str | None = Field(
        None, description="Opaque token held by the one durable delivery claimant"
    )
    claimed_at: datetime | None = Field(
        None, description="Timestamp of the current or most recent delivery claim"
    )
    superseded_by_message_id: int | None = Field(
        None,
        description=(
            "id of the exact PENDING row (from the same sender/receiver pair) that "
            "durably superseded this one, if any (correction-1038)."
        ),
    )
