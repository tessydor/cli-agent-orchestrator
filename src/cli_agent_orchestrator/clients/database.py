"""Minimal database client with only terminal metadata."""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import (
    DDL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, declarative_base, sessionmaker

from cli_agent_orchestrator.constants import DATABASE_URL, DB_DIR, DEFAULT_PROVIDER
from cli_agent_orchestrator.models.assigned_worker import (
    AssignedWorkerCallback,
    AssignedWorkerIntegrityError,
    AssignmentLifecycle,
    CompletionDeliveryState,
    CompletionReceiverState,
    callback_routing_digest,
    canonical_callback_text,
    format_server_completion_message,
)
from cli_agent_orchestrator.models.flow import Flow
from cli_agent_orchestrator.models.inbox import (
    InboxMessage,
    InboxMessageOrigin,
    MessageStatus,
)

logger = logging.getLogger(__name__)

Base: Any = declarative_base()


class TerminalModel(Base):
    """SQLAlchemy model for terminal metadata only."""

    __tablename__ = "terminals"

    id = Column(String, primary_key=True)  # "abc123ef"
    tmux_session = Column(String, nullable=False)  # "cao-session-name"
    tmux_window = Column(String, nullable=False)  # "window-name"
    provider = Column(String, nullable=False)  # "kiro_cli", "claude_code"
    agent_profile = Column(String)  # "developer", "reviewer" (optional)
    working_directory = Column(String, nullable=True)  # launch-time cwd (optional)
    allowed_tools = Column(String, nullable=True)  # JSON-encoded list of CAO tool names
    shell_command = Column(String, nullable=True)  # shell process name captured before kiro launch
    caller_id = Column(String, nullable=True)  # terminal that created this one (callback target)
    engine = Column(String, nullable=True)  # resolved Kiro engine; NULL for legacy/non-Kiro rows
    # Ordered, general-to-specific array of strings (JSON-encoded), e.g.
    # '["tenant_1", "project_5", "folder_12"]'. CAO only does ordered-prefix
    # matching (list_siblings); consumers own what the levels mean (#432).
    group = Column(Text, nullable=True)
    # Free-form JSON (JSON-encoded dict), consumer-defined, no fixed schema.
    # Python attribute is ``metadata_json`` (not ``metadata``) because
    # SQLAlchemy's declarative Base reserves ``.metadata`` for the schema
    # MetaData object on every mapped class; the DB column itself is still
    # literally named "metadata" per #432's design.
    metadata_json = Column("metadata", Text, nullable=True)
    last_active = Column(DateTime, default=datetime.now)


class InboxModel(Base):
    """SQLAlchemy model for inbox messages."""

    __tablename__ = "inbox"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sender_id = Column(String, nullable=False)
    receiver_id = Column(String, nullable=False)
    message = Column(String, nullable=False)
    status = Column(String, nullable=False)  # MessageStatus enum value
    # Origin and assignment linkage let successful assigned-worker completion
    # delivery suppress only an equivalent explicit final callback.  Legacy and
    # unrelated intermediate messages remain independent inbox rows.
    origin = Column(
        String,
        nullable=False,
        default=InboxMessageOrigin.LEGACY.value,
        # Keep a freshly created V1 database writable by a rolled-back 2.4.1
        # server, whose INSERT statements do not name this additive column.
        server_default=InboxMessageOrigin.LEGACY.value,
    )
    assignment_id = Column(String, nullable=True)
    # Nullable for legacy/explicit rows.  SQLite permits multiple NULLs in a
    # unique index, while server-generated callbacks use a stable non-NULL key.
    idempotency_key = Column(String, nullable=True)
    # Delivery uses an atomic PENDING -> DELIVERING compare-and-set.  Only the
    # owner of this opaque token may resolve the claim after terminal paste.
    claim_token = Column(String, nullable=True)
    claimed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (Index("uq_inbox_idempotency_key", "idempotency_key", unique=True),)


class AssignedWorkerCallbackModel(Base):
    """Durable successful-completion callback state for one assigned worker.

    No foreign key points at ``terminals``: worker and caller rows are operational
    records that may be deleted, while the completion report must remain available
    for manual recovery.  Identity/routing fields are written once at assignment
    creation and no update function below mutates them.
    """

    __tablename__ = "assigned_worker_callbacks"

    assignment_id = Column(String, primary_key=True)
    completion_id = Column(String, nullable=False)
    worker_terminal_id = Column(String, nullable=False)
    caller_id = Column(String, nullable=False)
    routing_digest = Column(String, nullable=False)
    lifecycle = Column(String, nullable=False, default=AssignmentLifecycle.ASSIGNED.value)
    delivery_state = Column(String, nullable=False, default=CompletionDeliveryState.NOT_READY.value)
    receiver_state = Column(String, nullable=False, default=CompletionReceiverState.UNKNOWN.value)
    final_result = Column(Text, nullable=True)
    final_result_sha256 = Column(String, nullable=True)
    result_reference = Column(String, nullable=True)
    inbox_message_id = Column(Integer, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    dispatched_at = Column(DateTime, nullable=True)
    completion_observed_at = Column(DateTime, nullable=True)
    captured_at = Column(DateTime, nullable=True)
    first_attempt_at = Column(DateTime, nullable=True)
    last_attempt_at = Column(DateTime, nullable=True)
    enqueued_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    terminal_error_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)

    __table_args__ = (
        Index("uq_assigned_worker_callback_completion_id", "completion_id", unique=True),
        Index("uq_assigned_worker_callback_worker", "worker_terminal_id", unique=True),
        Index("idx_assigned_worker_callback_delivery", "delivery_state"),
    )


_IMMUTABLE_CALLBACK_ROUTE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_route_immutable
BEFORE UPDATE OF assignment_id, completion_id, worker_terminal_id, caller_id, routing_digest
ON assigned_worker_callbacks
WHEN NEW.assignment_id IS NOT OLD.assignment_id
  OR NEW.completion_id IS NOT OLD.completion_id
  OR NEW.worker_terminal_id IS NOT OLD.worker_terminal_id
  OR NEW.caller_id IS NOT OLD.caller_id
  OR NEW.routing_digest IS NOT OLD.routing_digest
BEGIN
  SELECT RAISE(ABORT, 'assigned-worker callback routing is immutable');
END
"""

_IMMUTABLE_CALLBACK_RESULT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_result_immutable
BEFORE UPDATE OF final_result, final_result_sha256, result_reference
ON assigned_worker_callbacks
WHEN (OLD.final_result IS NOT NULL
   OR OLD.final_result_sha256 IS NOT NULL
   OR OLD.result_reference IS NOT NULL)
 AND (
      NEW.final_result IS NOT OLD.final_result
   OR NEW.final_result_sha256 IS NOT OLD.final_result_sha256
   OR NEW.result_reference IS NOT OLD.result_reference
 )
BEGIN
  SELECT RAISE(ABORT, 'captured assigned-worker result is immutable');
END
"""

_IMMUTABLE_CALLBACK_LINK_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_link_immutable
BEFORE UPDATE OF inbox_message_id ON assigned_worker_callbacks
WHEN OLD.inbox_message_id IS NOT NULL
 AND NEW.inbox_message_id IS NOT OLD.inbox_message_id
BEGIN
  SELECT RAISE(ABORT, 'assigned-worker callback inbox link is immutable once set');
END
"""

_RETAIN_CALLBACK_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_callback_retain
BEFORE DELETE ON assigned_worker_callbacks
BEGIN
  SELECT RAISE(ABORT, 'assigned-worker callback audit/report must be retained');
END
"""

_RETAIN_UNCLASSIFIED_WORKER_TERMINAL_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_terminal_recovery_retain
BEFORE DELETE ON terminals
WHEN EXISTS (
  SELECT 1 FROM assigned_worker_callbacks callback
  WHERE callback.worker_terminal_id = OLD.id
    AND callback.lifecycle IN ('assigned', 'dispatched', 'unresolved')
)
BEGIN
  SELECT RAISE(ABORT, 'unclassified assigned-worker terminal must be retained');
END
"""

_REQUIRE_RECEIVER_DELETE_AUDIT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_receiver_delete_audit
BEFORE DELETE ON terminals
WHEN EXISTS (
  SELECT 1 FROM assigned_worker_callbacks callback
  WHERE callback.caller_id = OLD.id
    AND callback.lifecycle IN ('assigned', 'dispatched', 'unresolved', 'completed')
    AND callback.receiver_state != 'deleted'
    AND callback.delivery_state != 'terminal_error'
)
BEGIN
  SELECT RAISE(ABORT, 'assigned-worker receiver deletion requires callback audit');
END
"""

_IMMUTABLE_LINKED_INBOX_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_inbox_evidence_immutable
BEFORE UPDATE OF id, sender_id, receiver_id, message, origin, assignment_id, idempotency_key
ON inbox
WHEN OLD.origin = 'server_completion'
 OR EXISTS (
  SELECT 1 FROM assigned_worker_callbacks callback
  WHERE callback.inbox_message_id = OLD.id
)
BEGIN
  SELECT RAISE(ABORT, 'linked assigned-worker inbox evidence is immutable');
END
"""

_RETAIN_LINKED_INBOX_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_inbox_evidence_retain
BEFORE DELETE ON inbox
WHEN EXISTS (
  SELECT 1 FROM assigned_worker_callbacks callback
  WHERE callback.inbox_message_id = OLD.id
)
BEGIN
  SELECT RAISE(ABORT, 'linked assigned-worker inbox evidence must be retained');
END
"""

_RETAIN_SERVER_INBOX_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_assigned_worker_server_inbox_retain
BEFORE DELETE ON inbox
WHEN OLD.origin = 'server_completion'
BEGIN
  SELECT RAISE(ABORT, 'assigned-worker server inbox evidence must be retained');
END
"""

_RETAIN_DELIVERED_INBOX_STATUS_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_delivered_inbox_status_immutable
BEFORE UPDATE OF status ON inbox
WHEN OLD.status = 'delivered' AND NEW.status != 'delivered'
 AND EXISTS (
  SELECT 1 FROM assigned_worker_callbacks callback
  WHERE callback.inbox_message_id = OLD.id
 )
BEGIN
  SELECT RAISE(ABORT, 'delivered inbox status is immutable');
END
"""

# Fresh databases receive the same guards as migrated databases.  The callback
# table is declared after inbox, so both referenced tables exist at after_create.
for _callback_trigger in (
    _IMMUTABLE_CALLBACK_ROUTE_TRIGGER,
    _IMMUTABLE_CALLBACK_RESULT_TRIGGER,
    _IMMUTABLE_CALLBACK_LINK_TRIGGER,
    _RETAIN_CALLBACK_TRIGGER,
    _RETAIN_UNCLASSIFIED_WORKER_TERMINAL_TRIGGER,
    _REQUIRE_RECEIVER_DELETE_AUDIT_TRIGGER,
    _IMMUTABLE_LINKED_INBOX_TRIGGER,
    _RETAIN_LINKED_INBOX_TRIGGER,
    _RETAIN_SERVER_INBOX_TRIGGER,
    _RETAIN_DELIVERED_INBOX_STATUS_TRIGGER,
):
    event.listen(AssignedWorkerCallbackModel.__table__, "after_create", DDL(_callback_trigger))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryMetadataModel(Base):
    """SQLAlchemy model for memory metadata (Phase 2 U1).

    SQLite is the source of truth for metadata queries; wiki markdown
    files remain the content store. Each row corresponds to exactly one
    wiki file on disk.
    """

    __tablename__ = "memory_metadata"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    memory_type = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    scope_id = Column(String, nullable=True)
    file_path = Column(String, nullable=False)
    tags = Column(String, nullable=False, default="")
    source_provider = Column(String, nullable=True)
    source_terminal_id = Column(String, nullable=True)
    token_estimate = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    # 3-factor scoring. ``access_count`` feeds the usage factor;
    # ``last_accessed_at`` backs a server-side rate-limit on increments. NOT
    # NULL DEFAULT 0 so existing rows read as "never recalled" without a
    # backfill. Migrated onto existing DBs by ``_migrate_add_access_count``.
    access_count = Column(Integer, nullable=False, default=0, server_default="0")
    last_accessed_at = Column(DateTime(timezone=True), nullable=True, default=None)
    # LLM wiki compilation. NULL = never LLM-compiled (pre-existing rows, or
    # every compile attempt fell back to append). Non-NULL = UTC timestamp of
    # the last successful compile.
    last_compiled_at = Column(DateTime(timezone=True), nullable=True, default=None)
    # Comma-separated sanitised keys of cross-referenced articles. NULL =
    # never computed (pre-existing rows or LLM error). ``""`` = computed, no
    # related found (success — distinct from NULL to avoid endless retries).
    # Practical max ≤ 256 bytes (3 keys × 60 chars + 2 commas). The CHECK
    # constraint applies on FRESH databases only — existing DBs rely on the
    # parse-side cap in ``_parse_related_keys``.
    related_keys = Column(Text, nullable=True, default=None)

    __table_args__ = (
        UniqueConstraint("key", "scope", "scope_id", name="uq_memory_key_scope"),
        CheckConstraint(
            "related_keys IS NULL OR length(related_keys) < 1024",
            name="ck_related_keys_length",
        ),
    )


# Relationship-store sentinel: ``memory_relationships.scope_id`` is NOT NULL and
# stores this value for global/federated scope. SQLite treats ``NULL != NULL``
# in a UNIQUE index, so a nullable scope_id would make the dedup index (and thus
# ``INSERT ... ON CONFLICT``) inert for global scope — silently duplicating
# exactly the edges hardest to notice. Storing a NOT-NULL sentinel keeps the
# dedup tuple total. ``""`` cannot collide with a real sanitized scope_id
# (``MemoryService._sanitize_key``/``_sanitize_scope_id`` never yield empty —
# the latter returns ``"unknown"``). This sentinel is scoped to the
# ``memory_relationships`` table ONLY; ``MemoryMetadataModel.scope_id`` remains
# genuinely nullable (stores real NULL for global), so cross-table endpoint
# checks against it use logical ``None`` + ``.is_(None)`` (see the relationship
# service), never this sentinel.
RELATIONSHIP_SCOPE_ID_SENTINEL = ""


class MemoryRelationshipModel(Base):
    """SQLAlchemy model for a typed, durable memory relationship edge (issue #511).

    The authoritative relationship store that replaces the lossy
    ``memory_metadata.related_keys`` text column. One row per typed edge between
    two memory keys in the same ``(scope, scope_id)``. Written and read ONLY
    through ``MemoryRelationshipService`` — no other component issues SQL against
    this table (FR-2.1 single-boundary invariant).

    ``related_keys`` on ``MemoryMetadataModel`` is retained UNCHANGED as the
    compiler's computation-state marker (NULL = never computed/error, ``""`` =
    computed-empty) and is NOT modified or retired by this table (retirement is a
    separate, later change gated on a loss-free proof).
    """

    __tablename__ = "memory_relationships"

    # Application-generated uuid4 string PK, matching ``MemoryMetadataModel.id``
    # (str(uuid4())). API-stable identifier exposed in mutation responses.
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scope = Column(String, nullable=False)
    # NOT NULL: sentinel RELATIONSHIP_SCOPE_ID_SENTINEL ("") for global/federated
    # so the dedup UNIQUE index is total (see the sentinel comment above).
    scope_id = Column(String, nullable=False)
    source_key = Column(String, nullable=False)
    target_key = Column(String, nullable=False)
    # Closed taxonomy reusing the graph EdgeType values.
    type = Column(String, nullable=False)  # relates_to | contradiction | supersedes
    # compiler | wiki_lint | human | legacy_related_keys | external_import(reserved)
    origin = Column(String, nullable=False)
    # active | proposal | rejected | superseded | deleted (auditable soft-delete)
    status = Column(String, nullable=False, default="active")
    # Optional evidence metadata. NULL = no evidence (NEVER fabricated / coerced
    # to 0); a stored value is a validated REAL in [0, 1].
    confidence = Column(Float, nullable=True, default=None)
    # Optional ordering hint (e.g. legacy related_keys position). NULL if none.
    rank = Column(Integer, nullable=True, default=None)
    # Bounded JSON blob. NULL if none; the CHECK caps FRESH DBs, the service
    # caps existing DBs (mirrors the ck_related_keys_length precedent).
    attributes_json = Column(Text, nullable=True, default=None)
    # The source memory's updated_at at write time; basis for staleness
    # detection (an edge is stale when this predates the source's current
    # updated_at). NULL when unknown.
    source_updated_at = Column(DateTime(timezone=True), nullable=True, default=None)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        # Dedup: differing type or origin coexist as distinct rows (multi-edge +
        # provenance-aware coexistence); a repeat of the same tuple upserts.
        # Every column is non-NULL (scope_id sentinel), so the index and
        # ON CONFLICT fire for ALL scopes including global.
        UniqueConstraint(
            "scope",
            "scope_id",
            "source_key",
            "target_key",
            "type",
            "origin",
            name="uq_memory_rel",
        ),
        # FRESH-DB CHECKs only (SQLite cannot retro-add a CHECK); the service
        # validates confidence range and attributes size on existing DBs.
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_memory_rel_confidence_range",
        ),
        CheckConstraint(
            "attributes_json IS NULL OR length(attributes_json) <= 2048",
            name="ck_memory_rel_attributes_size",
        ),
    )


class ProjectAliasModel(Base):
    """SQLAlchemy model for project identity aliases (Phase 2.5 U6).

    Maps historical/alternate project identifiers (cwd hashes, manual labels)
    to a canonical ``project_id`` so memory recall survives directory rename
    and worktree layouts.
    """

    __tablename__ = "project_aliases"

    # ``alias`` is the sole primary key: an alias maps to exactly one canonical
    # project_id, so reverse lookups (get_project_id_by_alias) are stable. A
    # cwd-hash first resolved via an override and later via its git remote
    # upserts the same row rather than creating a second, ambiguous mapping.
    alias = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False)  # "git_remote" | "cwd_hash" | "manual"
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class WorkflowOutcomeModel(Base):
    """SQLAlchemy model for workflow outcome records (self-learning Phase 1).

    One row per reported outcome of a unit of agent work (a workflow step,
    a package conversion, a review round). Outcomes are the raw signal the
    retrospector agent distills into memory lessons — they carry short
    labels and notes, never transcripts or file contents.
    """

    __tablename__ = "workflow_outcomes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String, nullable=False)
    workflow_name = Column(String, nullable=True)  # optional grouping label
    task_label = Column(String, nullable=False)  # e.g. "convert package X"
    agent_profile = Column(String, nullable=True)  # profile that did the work
    source_terminal_id = Column(String, nullable=True)
    success = Column(Boolean, nullable=False)
    score = Column(Integer, nullable=True)  # optional 0-100 metric
    friction_notes = Column(Text, nullable=False, default="")  # short, content-free
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class FlowModel(Base):
    """SQLAlchemy model for flow metadata."""

    __tablename__ = "flows"

    name = Column(String, primary_key=True)
    file_path = Column(String, nullable=False)
    schedule = Column(String, nullable=False)
    agent_profile = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    script = Column(String, nullable=True)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    enabled = Column(Boolean, default=True)


def _ensure_db_dir() -> None:
    """Create the DB dir owner-only (0o700).

    The DB stores sensitive data (workflow spec_snapshot carries full prompt
    bodies + inputs_json), so the dir is owner-only — the same posture as
    claude_code prompt files (0o600) and the audit log (0o700/0o600). mkdir's
    mode is ignored when the dir already exists (exist_ok) and is masked by
    umask on creation — the chmod enforces 0o700 in both cases, best-effort.
    """
    DB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(DB_DIR, 0o700)
    except OSError as e:
        logger.warning(f"Could not restrict DB dir permissions on {DB_DIR}: {e}")


# Module-level singletons
_ensure_db_dir()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables and apply schema migrations."""
    _migrate_project_aliases_schema()
    Base.metadata.create_all(bind=engine)
    _restrict_db_file_permissions()
    _migrate_terminals_schema()
    _migrate_inbox_callback_schema()
    _migrate_assigned_worker_integrity_schema()
    _migrate_memory_indexes()
    _migrate_add_access_count()
    _migrate_add_last_compiled_at()
    _migrate_add_related_keys()
    _migrate_workflow_index()
    _migrate_workflow_run()
    _migrate_workflow_run_indexes()
    _migrate_workflow_run_step()
    _migrate_workflow_outcome_indexes()
    _migrate_workflow_run_event()
    _migrate_workflow_run_seq()
    # Appended LAST (issue #511). Disjoint from the workflow_run* tables that
    # #504 also migrates, so registry order is immaterial — never reorder the
    # entries above.
    _migrate_memory_relationships()


def _restrict_db_file_permissions() -> None:
    """Chmod the SQLite file (+ -wal/-shm siblings if present) to 0o600.

    The DB persists sensitive data (workflow spec_snapshot prompt bodies,
    inputs_json), matching the owner-only posture of prompt files and the audit
    log. Called after ``create_all`` so the file exists. Best-effort: a chmod
    failure (exotic filesystems) degrades permissions only, never blocks startup.
    """
    from cli_agent_orchestrator.constants import DATABASE_FILE

    for path in (
        DATABASE_FILE,
        DATABASE_FILE.with_name(DATABASE_FILE.name + "-wal"),
        DATABASE_FILE.with_name(DATABASE_FILE.name + "-shm"),
    ):
        if not path.exists():
            continue
        try:
            os.chmod(path, 0o600)
        except OSError as e:
            logger.warning(f"Could not restrict DB file permissions on {path}: {e}")


def _migrate_project_aliases_schema() -> None:
    """Rebuild project_aliases if it predates the alias-only primary key.

    The table originally used a composite PK ``(project_id, alias)``, which
    allowed one alias to map to several project_ids and made reverse lookups
    nondeterministic. The new schema keys on ``alias`` alone. SQLite cannot
    alter a primary key in place, so drop and recreate. The table is an
    opportunistic identity cache rebuilt by ``resolve_project_id`` on demand,
    so dropping rows is safe. Runs before ``create_all`` so the fresh schema
    is created with the new PK.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master " "WHERE type='table' AND name='project_aliases'"
            ).fetchone()
            if row is None:
                return  # table doesn't exist yet — create_all builds it fresh
            cols = conn.execute("PRAGMA table_info(project_aliases)").fetchall()
            # PRAGMA returns rows: (cid, name, type, notnull, dflt_value, pk).
            # In the legacy schema both project_id and alias have pk>0; in the
            # new schema only alias does.
            pk_cols = {c[1] for c in cols if c[5]}
            if pk_cols != {"alias"}:
                conn.execute("DROP TABLE project_aliases")
                conn.commit()
                logger.info("Migration: rebuilt project_aliases with alias-only primary key")
    except Exception as e:
        logger.debug(f"project_aliases migration skipped: {e}")


def _migrate_memory_indexes() -> None:
    """Add explicit indexes on memory_metadata for query performance."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_metadata (scope, scope_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_metadata (updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_metadata (memory_type)"
            )
    except Exception as e:
        logger.debug(f"Memory index migration skipped: {e}")


def _migrate_add_access_count() -> None:
    """Add access_count and last_accessed_at columns to memory_metadata if missing.

    Idempotent: PRAGMA table_info gate, ALTER TABLE ADD COLUMN only
    when missing. Fresh DBs already have the columns from
    ``Base.metadata.create_all``. Existing rows get ``0`` / ``NULL`` — the
    correct values for "never recalled".
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_metadata)")
            columns = {row[1] for row in cursor.fetchall()}
            if "access_count" not in columns:
                conn.execute(
                    "ALTER TABLE memory_metadata ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
                )
                logger.info("Migration: added access_count column to memory_metadata")
            if "last_accessed_at" not in columns:
                conn.execute("ALTER TABLE memory_metadata ADD COLUMN last_accessed_at DATETIME")
                logger.info("Migration: added last_accessed_at column to memory_metadata")
    except Exception as e:
        logger.debug(f"Migration check for access_count failed: {e}")


def _migrate_add_last_compiled_at() -> None:
    """Add last_compiled_at column to memory_metadata if missing.

    Idempotent: skipped on fresh DBs (the column ships in the model) and on
    repeated runs. Existing Phase 1/2 rows get NULL — correct, since they were
    never LLM-compiled.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_metadata)")
            columns = {row[1] for row in cursor.fetchall()}
            if "last_compiled_at" not in columns:
                conn.execute("ALTER TABLE memory_metadata ADD COLUMN last_compiled_at DATETIME")
                logger.info("Migration: added last_compiled_at column to memory_metadata")
    except Exception as e:
        logger.debug(f"Migration check for last_compiled_at failed: {e}")


def _migrate_add_related_keys() -> None:
    """Add related_keys column to memory_metadata if missing.

    Reuses the idempotent ALTER pattern: PRAGMA table_info gate, ALTER TABLE
    ADD COLUMN only when missing. The CHECK(length < 1024) constraint applies
    to FRESH DBs only — adding a CHECK to an existing SQLite table requires a
    full table rebuild we deliberately avoid. Existing DBs rely on the
    parse-side 1024-byte cap in ``_parse_related_keys``.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_metadata)")
            columns = {row[1] for row in cursor.fetchall()}
            if "related_keys" not in columns:
                conn.execute("ALTER TABLE memory_metadata ADD COLUMN related_keys TEXT")
                logger.info("Migration: added related_keys column to memory_metadata")
    except Exception as e:
        logger.debug(f"Migration check for related_keys failed: {e}")


def _migrate_memory_relationships() -> None:
    """Create the ``memory_relationships`` table + indexes and backfill legacy
    links (issue #511). Appended LAST to the ``init_db()`` registry.

    Idempotent, zero-arg, self-connecting — mirrors the existing migrators.
    Failure is logged at debug and never propagated (a missing table is
    recoverable; the service degrades). ``CREATE TABLE IF NOT EXISTS`` covers
    existing DBs where ``Base.metadata.create_all`` (which builds the model with
    its CHECK constraints on fresh DBs) has already run or will run — the same
    fresh-vs-existing split the codebase uses for ``related_keys``.

    Disjoint from the ``workflow_run*`` tables (#504); registry order is
    immaterial.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_relationships ("
                "id TEXT PRIMARY KEY, "
                "scope TEXT NOT NULL, "
                "scope_id TEXT NOT NULL, "
                "source_key TEXT NOT NULL, "
                "target_key TEXT NOT NULL, "
                "type TEXT NOT NULL, "
                "origin TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'active', "
                "confidence REAL, "
                "rank INTEGER, "
                "attributes_json TEXT, "
                "source_updated_at DATETIME, "
                "created_at DATETIME, "
                "updated_at DATETIME"
                ")"
            )
            # Dedup UNIQUE index — total because scope_id is NOT NULL (sentinel),
            # so ON CONFLICT fires for all scopes including global.
            #
            # ACCEPTED REDUNDANCY on a FRESH db (human review, PR #524): there,
            # create_all() has already satisfied the model's UniqueConstraint via
            # an unnamed sqlite_autoindex, so this statement adds a SECOND index
            # over identical columns (the name matches the constraint, but SQLite
            # does not treat a table-level UNIQUE as a named index, so
            # IF NOT EXISTS does not suppress it). Kept deliberately: this
            # migrator must remain zero-arg and idempotent for EXISTING dbs,
            # where CREATE TABLE IF NOT EXISTS is a no-op and this is the ONLY
            # thing that establishes the dedup index that replace_set/create rely
            # on. Making it fresh-db-aware would mean probing pragma index_list
            # and branching — more moving parts in a path whose failure mode is
            # silent duplicate edges. The cost is one extra index on new
            # installs: some write amplification and disk, no correctness or
            # query-plan impact.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_rel ON memory_relationships "
                "(scope, scope_id, source_key, target_key, type, origin)"
            )
            # Lookup index for the common (scope, scope_id, source_key) read path.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_rel_lookup ON memory_relationships "
                "(scope, scope_id, source_key)"
            )
            conn.commit()
            _backfill_legacy_related_keys(conn)
    except Exception as e:
        logger.debug(f"memory_relationships migration skipped: {e}")


def _backfill_legacy_related_keys(conn: Any) -> None:
    """One-time, idempotent backfill of ``memory_metadata.related_keys`` into
    ``memory_relationships`` as ``type=relates_to, origin=legacy_related_keys,
    status=active, confidence=NULL`` rows (issue #511, FR-1.4/FR-1.5).

    - Gated per source memory: if any ``legacy_related_keys`` row already exists
      for that ``(scope, scope_id, source_key)``, the source is skipped, so
      re-running ``init_db()`` is a no-op (idempotent).
    - ``related_keys IS NULL`` or ``""`` yields zero rows (never-computed /
      computed-empty carry no edge). The NULL-vs-"" marker stays on
      ``related_keys`` UNCHANGED — this backfill only READS it (ADR-4).
    - ``confidence`` is always NULL (never fabricated — NFR-2.1). Order is
      preserved as ``rank``.
    - A target that no longer resolves to an in-scope memory, a self-link, or a
      key that fails the sanitiser is REPORTED (logged) and NOT written active
      (FR-1.5) — never silently activated.
    - ``scope_id`` is normalised to the sentinel ``""`` for global/federated so
      the dedup index is total.

    Best-effort: any failure is logged at debug and never propagated (the
    service can compute relationships later; a partial backfill is safe because
    the per-source gate resumes cleanly).
    """
    # Lazy import to avoid a circular import (memory_service imports database).
    try:
        from cli_agent_orchestrator.services.memory_service import MemoryService
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(f"backfill skipped (memory_service import): {e}")
        return

    now_iso = _utcnow().isoformat()
    reported: list[str] = []
    try:
        rows = conn.execute(
            "SELECT key, scope, scope_id, related_keys, updated_at "
            "FROM memory_metadata "
            "WHERE related_keys IS NOT NULL AND related_keys != ''"
        ).fetchall()
    except Exception as e:
        logger.debug(f"backfill skipped (memory_metadata read): {e}")
        return

    for key, scope, scope_id, related_keys, src_updated_at in rows:
        sentinel = scope_id if scope_id is not None else RELATIONSHIP_SCOPE_ID_SENTINEL
        # Per-source idempotency gate (exact = on the sentinel, never IS NULL).
        existing = conn.execute(
            "SELECT 1 FROM memory_relationships "
            "WHERE source_key = ? AND scope = ? AND scope_id = ? "
            "AND origin = 'legacy_related_keys' LIMIT 1",
            (key, scope, sentinel),
        ).fetchone()
        if existing is not None:
            continue

        targets = MemoryService._parse_related_keys(related_keys, scope)
        # Resolve which target keys actually exist in the SAME (scope, scope_id).
        for rank, target in enumerate(targets):
            if target == key:
                reported.append(f"{scope}/{scope_id}/{key}->{target}: self-link")
                continue
            # Endpoint existence against memory_metadata: scope_id is genuinely
            # nullable there (real NULL for global), so match logical NULL, NOT
            # the sentinel.
            if scope_id is None:
                found = conn.execute(
                    "SELECT 1 FROM memory_metadata "
                    "WHERE key = ? AND scope = ? AND scope_id IS NULL LIMIT 1",
                    (target, scope),
                ).fetchone()
            else:
                found = conn.execute(
                    "SELECT 1 FROM memory_metadata "
                    "WHERE key = ? AND scope = ? AND scope_id = ? LIMIT 1",
                    (target, scope, scope_id),
                ).fetchone()
            if found is None:
                reported.append(f"{scope}/{scope_id}/{key}->{target}: dangling")
                continue
            try:
                conn.execute(
                    "INSERT INTO memory_relationships "
                    "(id, scope, scope_id, source_key, target_key, type, origin, "
                    "status, confidence, rank, attributes_json, source_updated_at, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'relates_to', 'legacy_related_keys', "
                    "'active', NULL, ?, NULL, ?, ?, ?) "
                    "ON CONFLICT (scope, scope_id, source_key, target_key, type, origin) "
                    "DO NOTHING",
                    (
                        str(uuid.uuid4()),
                        scope,
                        sentinel,
                        key,
                        target,
                        rank,
                        src_updated_at,
                        now_iso,
                        now_iso,
                    ),
                )
            except Exception as e:
                logger.debug(f"backfill insert skipped for {key}->{target}: {e}")
    try:
        conn.commit()
    except Exception:  # pragma: no cover
        pass
    if reported:
        logger.warning(
            "memory_relationships backfill reported %d stale/malformed legacy "
            "links (NOT activated): %s",
            len(reported),
            "; ".join(reported[:20]),
        )


def _migrate_workflow_index() -> None:
    """Create/upgrade the derived ``workflow_index`` table (issue #312, N2).

    The table is a **derived, non-authoritative** projection of the workflow
    spec YAML files on disk (B2-BR-2): it can be dropped and rebuilt
    byte-identically from the files alone (``rebuild_index_from_files``). It
    carries no run/execution state — runs and per-step state are N5/N6.

    Idempotent (``CREATE TABLE IF NOT EXISTS``), zero-arg and self-connecting —
    mirrors the existing ``_migrate_memory_indexes`` pattern. Failure is logged
    at debug and never propagated (a missing index table is recoverable: the
    next ``list`` rebuilds it).

    U5 additively widens ``step_count`` to nullable: script-tier rows carry
    NULL (step count is run-time-determined, unknowable at index time), while
    YAML rows keep populating an int. ``CREATE TABLE IF NOT EXISTS`` only
    covers fresh DBs — on a pre-U5 DB the column already exists as NOT NULL,
    and SQLite cannot ``ALTER COLUMN`` to relax a NOT NULL constraint in
    place. Same drop/rebuild precedent as ``_migrate_project_aliases_schema``:
    the table is fully derived, so dropping it is safe — the next ``list``
    rebuilds it from the workflow files on disk.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_index'"
            ).fetchone()
            if row is not None:
                cols = conn.execute("PRAGMA table_info(workflow_index)").fetchall()
                # PRAGMA row: (cid, name, type, notnull, dflt_value, pk).
                step_count_col = next((c for c in cols if c[1] == "step_count"), None)
                if step_count_col is not None and step_count_col[3]:  # notnull flag set
                    conn.execute("DROP TABLE workflow_index")
                    conn.commit()
                    logger.info(
                        "Migration: rebuilt workflow_index with nullable step_count "
                        "(dropped legacy table; rebuilt from workflow files on next list)"
                    )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_index ("
                "name TEXT PRIMARY KEY, "
                "source_path TEXT NOT NULL, "
                "mode TEXT NOT NULL, "
                "step_count INTEGER, "  # nullable: script-tier rows carry NULL
                "description TEXT NOT NULL DEFAULT '', "
                "indexed_at TEXT NOT NULL"
                ")"
            )
    except Exception as e:  # noqa: BLE001 — derived table; rebuilt on next list
        logger.debug(f"workflow_index migration skipped: {e}")


def _migrate_workflow_run() -> None:
    """Create the durable ``workflow_run`` journal table if missing (issue #312, N6).

    The run aggregate root: one row per run, keyed by ``run_id`` (E1,
    domain-entities). Per Q1=B this is the **source of truth** for run execution
    state; the Bolt-3 in-memory ``run_registry`` is a cache over it. No loop
    columns (``iteration_counter`` etc.) — deferred to N8 (Q4=B, B4-BR-12).

    Idempotent (``CREATE TABLE IF NOT EXISTS``), zero-arg and self-connecting —
    mirrors ``_migrate_workflow_index`` (B2, B4-BR-1). Failure is logged at debug
    and never propagated: a missing table is recoverable, the next write retries
    the path and the live run completes on the in-memory floor (B4-RD-4).

    U3 (issue #312, script-tier journal extension) additively appends two
    columns — ``tier`` and ``generation`` (E1, domain-entities) — via the same
    idempotent ``PRAGMA table_info`` gate used by ``_migrate_add_access_count`` /
    ``_migrate_add_related_keys``. Both default to values that make a pre-U3 /
    YAML row read identically to its pre-extension form (INV-1/INV-2): existing
    rows back-fill to ``tier='yaml'``, ``generation='1'``. ``generation`` is TEXT,
    not INTEGER, so it compares byte-identically against the env-var-transported
    string generation value (domain-entities B4 fix).
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_run ("
                "run_id TEXT PRIMARY KEY, "
                "workflow_name TEXT NOT NULL, "
                "spec_snapshot TEXT NOT NULL, "
                "inputs_json TEXT NOT NULL, "
                "state TEXT NOT NULL, "
                "current_step_id TEXT, "
                "started_at TEXT NOT NULL, "
                "finished_at TEXT"
                ")"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(workflow_run)")}
            if "tier" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run ADD COLUMN tier TEXT NOT NULL DEFAULT 'yaml'"
                )
                logger.info("Migration: added tier column to workflow_run")
            if "generation" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run ADD COLUMN generation TEXT NOT NULL DEFAULT '1'"
                )
                logger.info("Migration: added generation column to workflow_run")
    except Exception as e:  # noqa: BLE001 — derived/recoverable; logged at debug (B4-RD-4)
        logger.debug(f"workflow_run migration skipped: {e}")


def _migrate_workflow_run_step() -> None:
    """Create the durable ``workflow_run_step`` table if missing (issue #312, N6).

    Per-step durable state: one row per ``(run_id, step_id)`` (E2,
    domain-entities). ``reprompted``/``terminal_id`` are deliberately NOT
    journaled (F3) — they are in-memory-only and defaulted on rebuild. No
    ``which_guard_fired``/``iterations_run`` columns — N8 adds them via its own
    additive migrator (Q4=B, B4-BR-12).

    Idempotent, zero-arg, self-connecting; failure logged at debug and never
    propagated (B4-BR-1 / B4-RD-4), same precedent as ``_migrate_workflow_index``.

    U3 (issue #312, script-tier journal extension) additively appends
    ``call_fingerprint`` (E2, domain-entities) via the same idempotent
    ``PRAGMA table_info`` gate. Defaults to ``NULL`` so a pre-U3 / YAML row is
    indistinguishable from its pre-extension form (INV-1/INV-2); ``append_step``
    is the sole write path for the column (``update_step`` stays untouched — the
    fingerprint is set once, at the RUNNING insert).

    U1 (issue #504, event-log substrate) additively appends three nullable
    columns via the same PRAGMA-gated ``ALTER TABLE ADD COLUMN`` idiom:
    ``terminal_id`` (associated terminal), ``reprompted`` (reprompt flag), and
    ``error_kind`` (structured error kind). All default to ``NULL`` so a
    pre-U1 row reads back observably identical to its pre-extension form
    (additive-only, C-1/C-4). ``workflow_run`` itself is untouched.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_run_step ("
                "run_id TEXT NOT NULL, "
                "step_id TEXT NOT NULL, "
                "state TEXT NOT NULL, "
                "attempts INTEGER NOT NULL, "
                "output_json TEXT, "
                "error TEXT, "
                "updated_at TEXT NOT NULL, "
                "PRIMARY KEY (run_id, step_id)"
                ")"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(workflow_run_step)")}
            if "call_fingerprint" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run_step ADD COLUMN call_fingerprint TEXT DEFAULT NULL"
                )
                logger.info("Migration: added call_fingerprint column to workflow_run_step")
            if "terminal_id" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run_step ADD COLUMN terminal_id TEXT DEFAULT NULL"
                )
                logger.info("Migration: added terminal_id column to workflow_run_step")
            if "reprompted" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run_step ADD COLUMN reprompted INTEGER DEFAULT NULL"
                )
                logger.info("Migration: added reprompted column to workflow_run_step")
            if "error_kind" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_run_step ADD COLUMN error_kind TEXT DEFAULT NULL"
                )
                logger.info("Migration: added error_kind column to workflow_run_step")
    except Exception as e:  # noqa: BLE001 — derived/recoverable; logged at debug (B4-RD-4)
        logger.debug(f"workflow_run_step migration skipped: {e}")


def _migrate_workflow_outcome_indexes() -> None:
    """Add indexes on workflow_outcomes for retrospector queries.

    The table itself is created by ``Base.metadata.create_all`` (it ships in
    the model, so fresh and existing DBs both get it). Retrospection filters
    by session and by agent profile over a recency window — index both.
    Idempotent, self-connecting, failure logged at debug — mirrors
    ``_migrate_memory_indexes``.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcome_session "
                "ON workflow_outcomes (session_name, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcome_agent "
                "ON workflow_outcomes (agent_profile, created_at)"
            )
    except Exception as e:
        logger.debug(f"workflow_outcomes index migration skipped: {e}")


def _migrate_workflow_run_event() -> None:
    """Create the durable append-only ``workflow_run_event`` table if missing (issue #504, U1).

    The event log root: one row per emitted workflow domain event, keyed by the
    composite ``(run_id, seq)`` PRIMARY KEY (ADR-1, domain-entities). Per
    NFR-DUR-1 this table is the authoritative, append-only, versioned record of
    workflow execution — rows are inserted and never updated or reordered; ``seq``
    (a per-run monotonically increasing sequence) is the SOLE ordering authority,
    ``ts`` is display/duration only (BR-5). ``run_id``, ``seq``, ``event_type``,
    ``event_schema_version`` (FR-1.1) and ``ts`` are NOT NULL; the remaining
    columns are nullable and populated where applicable. ``iteration`` and
    ``which_guard_fired`` are RESERVED for a later deterministic-loops feature
    (FR-1.5) and stay NULL in the MVP.

    Idempotent (``CREATE TABLE IF NOT EXISTS``), zero-arg and self-connecting —
    mirrors ``_migrate_workflow_run`` (C-1/C-4, additive-only). Failure is logged
    at debug and never propagated: a missing table is recoverable, the next
    best-effort append retries the path.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_run_event ("
                "run_id TEXT NOT NULL, "
                "seq INTEGER NOT NULL, "
                "event_type TEXT NOT NULL, "
                "event_schema_version INTEGER NOT NULL, "
                "ts TEXT NOT NULL, "
                "step_id TEXT, "
                "attempt INTEGER, "
                "state TEXT, "
                "elapsed_ms INTEGER, "
                "provider TEXT, "
                "agent_profile TEXT, "
                "engine TEXT, "
                "terminal_id TEXT, "
                "terminal_offset_start INTEGER, "
                "terminal_offset_len INTEGER, "
                "error_kind TEXT, "
                "reason TEXT, "
                "validation_result TEXT, "
                "output_ref TEXT, "
                "iteration INTEGER, "
                "which_guard_fired TEXT, "
                "PRIMARY KEY (run_id, seq)"
                ")"
            )
    except Exception as e:  # noqa: BLE001 — derived/recoverable; logged at debug
        logger.debug(f"workflow_run_event migration skipped: {e}")


def _migrate_workflow_run_seq() -> None:
    """Create the durable ``workflow_run_seq`` high-water table if missing (issue #504, U1).

    One row per run: ``high_water`` records the highest per-run ``seq`` ever
    ALLOCATED (best-effort persisted before the matching event append), so a
    rebuild can resume strictly above any allocated slot even when its append was
    swallowed (BR-3). ``high_water`` advances monotonically (BR-11) and is NOT
    NULL; ``run_id`` is the PRIMARY KEY.

    Idempotent (``CREATE TABLE IF NOT EXISTS``), zero-arg and self-connecting —
    same additive-only posture as ``_migrate_workflow_run_event``. Failure is
    logged at debug and never propagated.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_run_seq ("
                "run_id TEXT PRIMARY KEY, "
                "high_water INTEGER NOT NULL"
                ")"
            )
    except Exception as e:  # noqa: BLE001 — derived/recoverable; logged at debug
        logger.debug(f"workflow_run_seq migration skipped: {e}")


def _migrate_workflow_run_indexes() -> None:
    """Add explicit indexes on ``workflow_run`` for list-query performance (U1, FR-3.2).

    Two single-column indexes serving the two shapes ``list_runs`` produces: the
    unfiltered newest-first list orders by ``started_at`` alone (served by
    ``idx_workflow_run_started_at``), and the state-filtered list narrows on
    ``state`` (served by ``idx_workflow_run_state``). Two single-column indexes
    cover both paths; a single composite ``(state, started_at)`` would not serve
    the unfiltered ``started_at``-only ordering (ADR-6, IR-1).

    Zero-arg, self-connecting, and idempotent — mirrors ``_migrate_memory_indexes``.
    Each statement uses ``CREATE INDEX IF NOT EXISTS`` so a second ``init_db()`` is
    a no-op (IR-2); no destructive migration, no Alembic (NFR-5). It creates only
    indexes, never columns — so the C-4 exact-column migration test is untouched
    (IR-4). Registered AFTER ``_migrate_workflow_run`` in ``init_db`` so the base
    table exists first. Failure is logged at debug and never raised: a missing
    index degrades to a table scan, not a crash (IR-3).
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_run_started_at "
                "ON workflow_run (started_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_run_state ON workflow_run (state)"
            )
    except Exception as e:  # noqa: BLE001 — missing index degrades to a scan (IR-3)
        logger.debug(f"workflow_run index migration skipped: {e}")


def _migrate_terminals_schema() -> None:
    """Add terminal metadata columns to existing SQLite databases."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        conn = sqlite3.connect(str(DATABASE_FILE))
        cursor = conn.execute("PRAGMA table_info(terminals)")
        columns = {row[1] for row in cursor.fetchall()}
        if "allowed_tools" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN allowed_tools TEXT")
            conn.commit()
            logger.info("Migration: added allowed_tools column to terminals table")
        if "shell_command" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN shell_command TEXT")
            conn.commit()
            logger.info("Migration: added shell_command column to terminals table")
        if "caller_id" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN caller_id TEXT")
            conn.commit()
            logger.info("Migration: added caller_id column to terminals table")
        if "engine" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN engine TEXT")
            conn.commit()
            logger.info("Migration: added engine column to terminals table")
        if "group" not in columns:
            # "group" is a SQL reserved word in some dialects but not SQLite;
            # quoted defensively so this ALTER survives if that ever changes.
            conn.execute('ALTER TABLE terminals ADD COLUMN "group" TEXT')
            conn.commit()
            logger.info("Migration: added group column to terminals table")
        if "metadata" not in columns:
            conn.execute('ALTER TABLE terminals ADD COLUMN "metadata" TEXT')
            conn.commit()
            logger.info("Migration: added metadata column to terminals table")
        if "working_directory" not in columns:
            conn.execute("ALTER TABLE terminals ADD COLUMN working_directory TEXT")
            conn.commit()
            logger.info("Migration: added working_directory column to terminals table")
        conn.close()
    except Exception as e:
        logger.warning(f"Migration check for terminals schema failed: {e}")


def _migrate_inbox_callback_schema() -> None:
    """Add callback audit/idempotency columns to an existing inbox table.

    ``Base.metadata.create_all`` creates ``assigned_worker_callbacks`` and the
    complete inbox schema for fresh installations.  SQLite does not add columns
    to an existing table, so upgrades use additive nullable/defaulted columns and
    a named unique index.  The migration is idempotent and preserves every legacy
    inbox row as ``origin='legacy'``.
    """
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inbox)")}
            if not columns:
                # A mocked/partial startup may not have created the table.  The
                # normal create_all path handles it on the next real startup.
                return
            if "origin" not in columns:
                conn.execute("ALTER TABLE inbox ADD COLUMN origin TEXT NOT NULL DEFAULT 'legacy'")
            if "assignment_id" not in columns:
                conn.execute("ALTER TABLE inbox ADD COLUMN assignment_id TEXT")
            if "idempotency_key" not in columns:
                conn.execute("ALTER TABLE inbox ADD COLUMN idempotency_key TEXT")
            if "claim_token" not in columns:
                conn.execute("ALTER TABLE inbox ADD COLUMN claim_token TEXT")
            if "claimed_at" not in columns:
                conn.execute("ALTER TABLE inbox ADD COLUMN claimed_at DATETIME")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_inbox_idempotency_key "
                "ON inbox (idempotency_key)"
            )
            conn.commit()
    except Exception as e:
        # Callback idempotency depends on this unique index.  Continuing with a
        # partially migrated inbox would turn duplicate completion events into
        # duplicate supervisor messages, so startup must fail closed.
        logger.error(f"Migration check for inbox callback schema failed: {e}")
        raise


def _migrate_assigned_worker_integrity_schema() -> None:
    """Anchor immutable routes and install fail-closed SQLite integrity guards."""
    import sqlite3

    from cli_agent_orchestrator.constants import DATABASE_FILE

    try:
        with sqlite3.connect(str(DATABASE_FILE)) as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(assigned_worker_callbacks)")
            }
            if not columns:
                return
            if "routing_digest" not in columns:
                conn.execute("ALTER TABLE assigned_worker_callbacks ADD COLUMN routing_digest TEXT")

            rows = conn.execute(
                "SELECT assignment_id, completion_id, worker_terminal_id, caller_id, "
                "routing_digest FROM assigned_worker_callbacks"
            ).fetchall()
            for assignment_id, completion_id, worker_terminal_id, caller_id, digest in rows:
                expected = callback_routing_digest(
                    assignment_id, completion_id, worker_terminal_id, caller_id
                )
                if digest is None:
                    conn.execute(
                        "UPDATE assigned_worker_callbacks SET routing_digest = ? "
                        "WHERE assignment_id = ?",
                        (expected, assignment_id),
                    )
                elif digest != expected:
                    raise AssignedWorkerIntegrityError(
                        f"Routing digest mismatch during migration for assignment {assignment_id}"
                    )

            callback_triggers = (
                _IMMUTABLE_CALLBACK_ROUTE_TRIGGER,
                _IMMUTABLE_CALLBACK_RESULT_TRIGGER,
                _IMMUTABLE_CALLBACK_LINK_TRIGGER,
                _RETAIN_CALLBACK_TRIGGER,
                _RETAIN_UNCLASSIFIED_WORKER_TERMINAL_TRIGGER,
                _REQUIRE_RECEIVER_DELETE_AUDIT_TRIGGER,
                _IMMUTABLE_LINKED_INBOX_TRIGGER,
                _RETAIN_LINKED_INBOX_TRIGGER,
                _RETAIN_SERVER_INBOX_TRIGGER,
                _RETAIN_DELIVERED_INBOX_STATUS_TRIGGER,
            )
            # Trigger definitions are versioned code, not merely presence
            # markers. Recreate every V1 guard so a database exercised by an
            # earlier draft cannot retain weaker result, route, or deletion
            # semantics behind CREATE IF NOT EXISTS.
            for trigger_name in (
                "trg_assigned_worker_route_immutable",
                "trg_assigned_worker_result_immutable",
                "trg_assigned_worker_link_immutable",
                "trg_assigned_worker_callback_retain",
                "trg_assigned_worker_terminal_recovery_retain",
                "trg_assigned_worker_receiver_delete_audit",
                "trg_assigned_worker_inbox_evidence_immutable",
                "trg_assigned_worker_inbox_evidence_retain",
                "trg_assigned_worker_server_inbox_retain",
                "trg_delivered_inbox_status_immutable",
            ):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
            for trigger_sql in callback_triggers:
                conn.execute(trigger_sql)
            conn.commit()
    except Exception as e:
        # Starting without these guards would make self-consistent route
        # rewriting indistinguishable from a legitimate assignment.
        logger.error(f"Assigned-worker integrity migration failed: {e}")
        raise


def create_terminal(
    terminal_id: str,
    tmux_session: str,
    tmux_window: str,
    provider: str,
    agent_profile: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    shell_command: Optional[str] = None,
    caller_id: Optional[str] = None,
    engine: Optional[str] = None,
    group: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    working_directory: Optional[str] = None,
    assignment_id: Optional[str] = None,
    completion_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create terminal metadata and, when supplied, its assignment atomically.

    The terminal and assigned-worker callback rows share one transaction.  A
    crash can therefore never expose an assigned worker whose immutable caller
    and completion identities were not persisted.
    """
    import json as _json

    if bool(assignment_id) != bool(completion_id):
        raise ValueError("assignment_id and completion_id must be supplied together")
    if assignment_id and not caller_id:
        raise ValueError("an assigned-worker callback requires caller_id")

    with SessionLocal() as db:
        terminal = TerminalModel(
            id=terminal_id,
            tmux_session=tmux_session,
            tmux_window=tmux_window,
            provider=provider,
            agent_profile=agent_profile,
            working_directory=working_directory,
            allowed_tools=_json.dumps(allowed_tools) if allowed_tools else None,
            shell_command=shell_command,
            caller_id=caller_id,
            engine=engine,
            group=_json.dumps(group) if group else None,
            metadata_json=_json.dumps(metadata) if metadata else None,
        )
        db.add(terminal)
        if assignment_id and completion_id:
            db.add(
                AssignedWorkerCallbackModel(
                    assignment_id=assignment_id,
                    completion_id=completion_id,
                    worker_terminal_id=terminal_id,
                    caller_id=caller_id,
                    routing_digest=callback_routing_digest(
                        assignment_id, completion_id, terminal_id, caller_id
                    ),
                    lifecycle=AssignmentLifecycle.ASSIGNED.value,
                    delivery_state=CompletionDeliveryState.NOT_READY.value,
                    receiver_state=CompletionReceiverState.UNKNOWN.value,
                )
            )
        db.commit()
        return {
            "id": terminal.id,
            "tmux_session": terminal.tmux_session,
            "tmux_window": terminal.tmux_window,
            "provider": terminal.provider,
            "agent_profile": terminal.agent_profile,
            "working_directory": terminal.working_directory,
            "allowed_tools": allowed_tools,
            "shell_command": terminal.shell_command,
            "caller_id": terminal.caller_id,
            "engine": terminal.engine,
            # Normalized the same way as what was actually stored (an empty
            # container is stored as NULL, same as omitted) -- self-ROAST
            # finding: echoing the raw `group`/`metadata` input here made
            # create_terminal(group=[]) return {"group": []} while an
            # immediately-following get_terminal_metadata() on the same row
            # returns {"group": None}, an API-consistency gap.
            "group": group if group else None,
            "metadata": metadata if metadata else None,
        }


def get_terminal_metadata(terminal_id: str) -> Optional[Dict[str, Any]]:
    """Get terminal metadata by ID."""
    import json as _json

    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if not terminal:
            logger.warning(f"Terminal metadata not found for terminal_id: {terminal_id}")
            return None
        logger.debug(
            f"Retrieved terminal metadata for {terminal_id}: provider={terminal.provider}, session={terminal.tmux_session}"
        )
        allowed_tools = _json.loads(terminal.allowed_tools) if terminal.allowed_tools else None
        group = _json.loads(terminal.group) if terminal.group else None
        metadata = _json.loads(terminal.metadata_json) if terminal.metadata_json else None
        return {
            "id": terminal.id,
            "tmux_session": terminal.tmux_session,
            "tmux_window": terminal.tmux_window,
            "provider": terminal.provider,
            "agent_profile": terminal.agent_profile,
            "working_directory": terminal.working_directory,
            "allowed_tools": allowed_tools,
            "shell_command": terminal.shell_command,
            "caller_id": terminal.caller_id,
            "engine": terminal.engine or ("v2" if terminal.provider == "kiro_cli" else None),
            "group": group,
            "metadata": metadata,
            "last_active": terminal.last_active,
        }


def update_terminal_group(terminal_id: str, group: Optional[List[str]]) -> bool:
    """Replace a terminal's group array. ``None``/``[]`` clears it (opts out of discovery)."""
    import json as _json

    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if not terminal:
            return False
        terminal.group = _json.dumps(group) if group else None
        db.commit()
        return True


def update_terminal_metadata(terminal_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
    """Replace a terminal's free-form metadata dict. ``None``/``{}`` clears it."""
    import json as _json

    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if not terminal:
            return False
        terminal.metadata_json = _json.dumps(metadata) if metadata else None
        db.commit()
        return True


def get_terminal_group(terminal_id: str) -> Optional[List[str]]:
    """Return a terminal's own group array, or None if unset or the terminal doesn't exist."""
    import json as _json

    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if not terminal or not terminal.group:
            return None
        return cast(List[str], _json.loads(terminal.group))


def list_siblings_by_group_prefix(
    caller_id: str,
    prefix: List[str],
    caller_session: Optional[str] = None,
    cross_session: bool = False,
) -> List[Dict[str, Any]]:
    """Return ``{id, group, metadata}`` for every OTHER terminal sharing ``prefix``.

    ``prefix`` is the caller's own group truncated to the (already-clamped)
    depth — this function does no clamping itself, it only matches. A
    candidate terminal with no group, or a group shorter than ``len(prefix)``,
    is excluded rather than compared partially or raising (#432).

    Session-scoped by default (issue #432 design discussion, tedswinyar +
    klabulan, 2026-07-17/18): ``caller_session`` (the caller's own
    ``tmux_session``) is an implicit, non-bypassable first filter ON TOP of
    the group-prefix match, unless ``cross_session=True`` is explicitly
    passed. Without this, two unrelated CAO sessions that happen to reuse
    the same ``group`` prefix (a naming collision, a copy-pasted template,
    two features that picked the same tenant/project id) would silently
    discover each other -- the same class of "implicitly-scoped state that
    turns out not to be" mistake cited in that discussion's incident
    history. Cross-session discovery is a legitimate use case and stays
    available, just opt-in rather than the unstated default.

    ``group`` is stored JSON-encoded (see ``TerminalModel.group``), so the
    query prefilters with a SQL ``LIKE`` prefix match on that encoding before
    loading/decoding candidate rows in Python (Copilot review, PR #433) —
    without it this scanned and JSON-decoded every grouped terminal on the
    server regardless of how narrow ``prefix`` is. Because ``json.dumps``
    closes each string element in a quote immediately, the encoded prefix
    (full array minus its trailing ``]``) can't false-positive match a
    longer sibling element that merely shares a text prefix (e.g. prefix
    element ``"project_5"`` vs. a sibling group containing ``"project_50"``
    — the sibling's extra ``0`` before its closing ``"`` breaks the SQL
    match). The exact Python-level comparison below is kept regardless, as
    the source of truth — the SQL match only narrows candidates (a prefilter
    defect can only cause a false negative here, i.e. a missed perf win,
    never a false positive / correctness or security regression).

    This SQL-level match assumes the stored ``group`` was encoded with the
    same ``json.dumps`` defaults used below (notably ``ensure_ascii=True``,
    the default) — true today of both write paths (``create_terminal`` and
    ``update_terminal_group``), which both use plain ``json.dumps(group)``.
    If either write path ever changes its encoding, this prefilter must
    change with it.

    A single row with corrupt ``group`` JSON (e.g. hand-edited DB, a future
    write-path bug) is logged and excluded rather than raising and failing
    discovery for every OTHER terminal in the same request (tedswinyar, PR
    #433 review). Corrupt ``metadata`` JSON on an otherwise-matching sibling
    is likewise logged and reported back as ``metadata=None`` -- the sibling
    itself is still real and discoverable, only its metadata is unreadable.
    """
    import json as _json

    depth = len(prefix)
    # Encode the prefix array and drop its trailing ']' so this matches both
    # a sibling group of the same length and a longer one that starts with
    # it, e.g. prefix ["a", "b"] -> '["a", "b"' matches '["a", "b"]' and
    # '["a", "b", "c"]'.
    like_prefix = _json.dumps(prefix)[:-1]
    with SessionLocal() as db:
        query = db.query(TerminalModel).filter(
            TerminalModel.id != caller_id,
            TerminalModel.group.isnot(None),
            TerminalModel.group.startswith(like_prefix, autoescape=True),
        )
        if not cross_session and caller_session is not None:
            query = query.filter(TerminalModel.tmux_session == caller_session)
        rows = query.all()
        siblings = []
        for row in rows:
            try:
                sibling_group = _json.loads(row.group)
                if not isinstance(sibling_group, list):
                    raise ValueError(f"decoded to {type(sibling_group).__name__}, expected list")
            except (TypeError, ValueError) as e:
                logger.warning(
                    "list_siblings_by_group_prefix: skipping terminal %s -- "
                    "corrupt group JSON (%s)",
                    row.id,
                    e,
                )
                continue
            if len(sibling_group) < depth:
                continue
            if sibling_group[:depth] == prefix:
                metadata = None
                if row.metadata_json:
                    try:
                        metadata = _json.loads(row.metadata_json)
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            "list_siblings_by_group_prefix: terminal %s has "
                            "corrupt metadata JSON (%s); returning it with "
                            "metadata=None",
                            row.id,
                            e,
                        )
                siblings.append(
                    {
                        "id": row.id,
                        "group": sibling_group,
                        "metadata": metadata,
                    }
                )
        return siblings


def list_terminals_by_session(tmux_session: str) -> List[Dict[str, Any]]:
    """List all terminals in a tmux session."""
    with SessionLocal() as db:
        terminals = db.query(TerminalModel).filter(TerminalModel.tmux_session == tmux_session).all()
        return [
            {
                "id": t.id,
                "tmux_session": t.tmux_session,
                "tmux_window": t.tmux_window,
                "provider": t.provider,
                "agent_profile": t.agent_profile,
                "working_directory": t.working_directory,
                "engine": t.engine or ("v2" if t.provider == "kiro_cli" else None),
                "last_active": t.last_active,
            }
            for t in terminals
        ]


def update_last_active(terminal_id: str) -> bool:
    """Update last active timestamp."""
    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if terminal:
            terminal.last_active = datetime.now()
            db.commit()
            return True
        return False


def update_terminal_shell_command(terminal_id: str, shell_command: str) -> bool:
    """Update the shell_command baseline for a terminal."""
    with SessionLocal() as db:
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if terminal:
            terminal.shell_command = shell_command
            db.commit()
            return True
        return False


def list_all_terminals() -> List[Dict[str, Any]]:
    """List all terminals."""
    with SessionLocal() as db:
        terminals = db.query(TerminalModel).all()
        return [
            {
                "id": t.id,
                "tmux_session": t.tmux_session,
                "tmux_window": t.tmux_window,
                "provider": t.provider,
                "agent_profile": t.agent_profile,
                "working_directory": t.working_directory,
                "engine": t.engine or ("v2" if t.provider == "kiro_cli" else None),
                "last_active": t.last_active,
            }
            for t in terminals
        ]


def list_pending_receiver_ids_by_provider(provider: str) -> List[str]:
    """List receiver terminal IDs with pending messages for a specific provider."""
    with SessionLocal() as db:
        rows = (
            db.query(InboxModel.receiver_id)
            .join(TerminalModel, TerminalModel.id == InboxModel.receiver_id)
            .filter(
                TerminalModel.provider == provider,
                InboxModel.status == MessageStatus.PENDING.value,
            )
            .distinct()
            .all()
        )
        return [row[0] for row in rows]


def list_pending_receiver_ids_older_than(min_age_seconds: int) -> List[str]:
    """List receiver terminal IDs whose messages have been PENDING too long.

    Returns the distinct receivers of any message still PENDING for longer than
    ``min_age_seconds``. Used by the inbox reconciliation sweep to find messages
    the immediate and watchdog delivery paths missed, without competing with
    them for freshly queued ones (issue #131).

    The join on ``terminals`` drops messages whose receiver terminal no longer
    exists, so the sweep does not keep retrying deliveries to deleted agents.

    ``created_at`` is stored local-naive (``InboxModel.created_at`` defaults to
    ``datetime.now``), so the cutoff uses ``datetime.now()`` to match — the same
    convention as the retention query in ``cleanup_service.cleanup_old_data``.
    """
    cutoff = datetime.now() - timedelta(seconds=min_age_seconds)
    with SessionLocal() as db:
        rows = (
            db.query(InboxModel.receiver_id)
            .join(TerminalModel, TerminalModel.id == InboxModel.receiver_id)
            .filter(
                InboxModel.status == MessageStatus.PENDING.value,
                InboxModel.created_at < cutoff,
            )
            .distinct()
            .all()
        )
        return [row[0] for row in rows]


def _audit_receiver_deletion(db: Any, terminal_id: str) -> None:
    """Record the true fate of callbacks routed to a deleted receiver."""
    callbacks = (
        db.query(AssignedWorkerCallbackModel)
        .filter(AssignedWorkerCallbackModel.caller_id == terminal_id)
        .all()
    )
    for callback in callbacks:
        _validate_assigned_worker_callback_row(db, callback)
        lifecycle = AssignmentLifecycle(callback.lifecycle)
        linked = None
        if callback.inbox_message_id is not None:
            linked = db.query(InboxModel).filter(InboxModel.id == callback.inbox_message_id).first()

        # A completed paste is durable evidence that deletion happened later;
        # preserve the successful historical state while recording the
        # receiver's true current disposition. This update also permits the
        # central delete through the raw-delete audit trigger below.
        if linked is not None and linked.status == MessageStatus.DELIVERED.value:
            callback.receiver_state = CompletionReceiverState.DELETED.value
            callback.last_error = (
                f"Persisted caller terminal {terminal_id} was deleted after callback paste; "
                "durable delivered evidence is preserved"
            )
            db.flush()
            _validate_assigned_worker_callback_row(db, callback)
            continue

        if lifecycle == AssignmentLifecycle.COMPLETED:
            if linked is not None and linked.status in (
                MessageStatus.PENDING.value,
                MessageStatus.DELIVERING.value,
            ):
                linked.status = MessageStatus.FAILED.value
                linked.claim_token = None
            callback.delivery_state = CompletionDeliveryState.TERMINAL_ERROR.value
            callback.receiver_state = CompletionReceiverState.DELETED.value
            callback.terminal_error_at = datetime.now()
            callback.last_error = (
                f"Persisted caller terminal {terminal_id} was deleted before callback paste; "
                "retained report requires manual recovery"
            )
        elif lifecycle in (
            AssignmentLifecycle.ASSIGNED,
            AssignmentLifecycle.DISPATCHED,
            AssignmentLifecycle.UNRESOLVED,
        ):
            # The task may still complete.  Keep it capture-reconcilable; once
            # captured, normal receiver classification records TERMINAL_ERROR.
            callback.receiver_state = CompletionReceiverState.DELETED.value
            if lifecycle != AssignmentLifecycle.UNRESOLVED:
                callback.last_error = (
                    f"Persisted caller terminal {terminal_id} was deleted before task completion"
                )
        db.flush()
        _validate_assigned_worker_callback_row(db, callback)


def delete_terminal(
    terminal_id: str,
    *,
    missing_backend: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Delete terminal metadata under the central report-preservation invariant.

    Direct deletion refuses any worker whose task outcome is not durably
    classified.  A caller that has positively established the backend handle is
    already gone may pass ``missing_backend=True`` plus an audit reason; that
    atomically records FAILED/TERMINAL_ERROR before deleting the stale row.
    """
    if missing_backend and not reason:
        raise ValueError("missing_backend deletion requires an auditable reason")
    with SessionLocal() as db:
        # Serialize receiver audit and terminal deletion with callback enqueue
        # and explicit-final equivalence transactions.
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        terminal = db.query(TerminalModel).filter(TerminalModel.id == terminal_id).first()
        if terminal is None:
            db.rollback()
            return False

        worker_callback = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.worker_terminal_id == terminal_id)
            .first()
        )
        if worker_callback is not None:
            _validate_assigned_worker_callback_row(db, worker_callback)
            lifecycle = AssignmentLifecycle(worker_callback.lifecycle)
            if lifecycle in (
                AssignmentLifecycle.ASSIGNED,
                AssignmentLifecycle.DISPATCHED,
                AssignmentLifecycle.UNRESOLVED,
            ):
                if not missing_backend:
                    db.rollback()
                    logger.warning(
                        "Refusing direct deletion of assigned worker %s with unclassified "
                        "lifecycle %s",
                        terminal_id,
                        lifecycle.value,
                    )
                    return False
                worker_callback.lifecycle = AssignmentLifecycle.FAILED.value
                worker_callback.delivery_state = CompletionDeliveryState.TERMINAL_ERROR.value
                worker_callback.receiver_state = CompletionReceiverState.UNKNOWN.value
                worker_callback.terminal_error_at = datetime.now()
                worker_callback.last_error = reason
                db.flush()
                _validate_assigned_worker_callback_row(db, worker_callback)

        _audit_receiver_deletion(db, terminal_id)
        db.delete(terminal)
        db.commit()
        return True


def delete_terminals_by_session(
    tmux_session: str,
    *,
    missing_backend: bool = False,
    reason: Optional[str] = None,
) -> int:
    """Delete session rows one-by-one through the central deletion invariant."""
    with SessionLocal() as db:
        terminal_ids = [
            row[0]
            for row in db.query(TerminalModel.id)
            .filter(TerminalModel.tmux_session == tmux_session)
            .all()
        ]
    deleted = 0
    for terminal_id in terminal_ids:
        if delete_terminal(
            terminal_id,
            missing_backend=missing_backend,
            reason=reason,
        ):
            deleted += 1
    return deleted


def _inbox_message_from_row(row: InboxModel) -> InboxMessage:
    """Convert an ORM inbox row, including legacy rows, to its read model."""
    raw_origin = getattr(row, "origin", None) or InboxMessageOrigin.LEGACY.value
    try:
        origin = InboxMessageOrigin(raw_origin)
    except ValueError:
        origin = InboxMessageOrigin.LEGACY
    return InboxMessage(
        id=row.id,
        sender_id=row.sender_id,
        receiver_id=row.receiver_id,
        message=row.message,
        status=MessageStatus(row.status),
        created_at=row.created_at,
        origin=origin,
        assignment_id=getattr(row, "assignment_id", None),
        idempotency_key=getattr(row, "idempotency_key", None),
        claim_token=getattr(row, "claim_token", None),
        claimed_at=getattr(row, "claimed_at", None),
    )


def create_inbox_message(
    sender_id: str,
    receiver_id: str,
    message: str,
    *,
    origin: InboxMessageOrigin = InboxMessageOrigin.LEGACY,
    assignment_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> InboxMessage:
    """Create one durable inbox row, serializing callback equivalence races.

    Explicit-final suppression and server-callback insertion share one SQLite
    ``BEGIN IMMEDIATE`` transaction.  Whichever ordering wins commits the sole
    callback evidence; the loser returns that row instead of inserting another
    supervisor-visible message.  Non-equivalent explicit traffic is unchanged.
    """

    def _matches_immutable_payload(existing: InboxModel) -> bool:
        return (
            cast(str, existing.sender_id) == sender_id
            and cast(str, existing.receiver_id) == receiver_id
            and cast(str, existing.message) == message
            and cast(Optional[str], existing.assignment_id) == assignment_id
            and cast(str, existing.origin) == origin.value
        )

    def _new_row(
        *,
        row_origin: InboxMessageOrigin = origin,
        row_assignment_id: Optional[str] = assignment_id,
        row_idempotency_key: Optional[str] = idempotency_key,
    ) -> InboxModel:
        row = InboxModel(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message=message,
            status=MessageStatus.PENDING.value,
            origin=row_origin.value,
            assignment_id=row_assignment_id,
            idempotency_key=row_idempotency_key,
        )
        db.add(row)
        db.flush()
        return row

    def _link_callback(callback: AssignedWorkerCallbackModel, inbox_row: InboxModel) -> None:
        now = datetime.now()
        callback.inbox_message_id = inbox_row.id
        callback.enqueued_at = callback.enqueued_at or inbox_row.created_at or now
        callback.acknowledged_at = callback.acknowledged_at or now
        callback.last_error = None
        callback.delivery_state = (
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value
            if inbox_row.origin == InboxMessageOrigin.EXPLICIT.value
            else CompletionDeliveryState.ACKNOWLEDGED.value
        )

    callback_write = origin in (
        InboxMessageOrigin.EXPLICIT,
        InboxMessageOrigin.SERVER_COMPLETION,
    )
    with SessionLocal() as db:
        if callback_write or idempotency_key is not None:
            # Acquire the SQLite writer slot before any equivalence read.  This
            # closes both explicit-first/server-first callback races and the
            # ordinary idempotency-key check/insert race.
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")

        if not db.query(TerminalModel).filter(TerminalModel.id == receiver_id).first():
            raise ValueError(f"Terminal '{receiver_id}' not found")

        callback: Optional[AssignedWorkerCallbackModel] = None
        if origin == InboxMessageOrigin.EXPLICIT and assignment_id is None:
            callback = (
                db.query(AssignedWorkerCallbackModel)
                .filter(
                    AssignedWorkerCallbackModel.worker_terminal_id == sender_id,
                    AssignedWorkerCallbackModel.caller_id == receiver_id,
                    AssignedWorkerCallbackModel.lifecycle.in_(
                        (
                            AssignmentLifecycle.ASSIGNED.value,
                            AssignmentLifecycle.DISPATCHED.value,
                            AssignmentLifecycle.COMPLETED.value,
                            AssignmentLifecycle.UNRESOLVED.value,
                        )
                    ),
                )
                .order_by(AssignedWorkerCallbackModel.created_at.desc())
                .first()
            )
            if callback is not None:
                _validate_assigned_worker_callback_row(db, callback)
                assignment_id = cast(str, callback.assignment_id)
        elif assignment_id is not None:
            callback = (
                db.query(AssignedWorkerCallbackModel)
                .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
                .first()
            )
            if callback is not None:
                _validate_assigned_worker_callback_row(db, callback)

        if origin == InboxMessageOrigin.EXPLICIT and callback is not None:
            equivalent_final = (
                callback.lifecycle == AssignmentLifecycle.COMPLETED.value
                and callback.final_result is not None
                and callback.delivery_state != CompletionDeliveryState.TERMINAL_ERROR.value
                and canonical_callback_text(message)
                == canonical_callback_text(callback.final_result)
            )
            if equivalent_final:
                # Server already won: satisfy the explicit API call with its
                # exact row.  No second explicit row becomes supervisor input.
                if callback.inbox_message_id is not None:
                    linked = (
                        db.query(InboxModel)
                        .filter(InboxModel.id == callback.inbox_message_id)
                        .first()
                    )
                    if linked is None:
                        raise _callback_integrity_error(callback, "linked inbox row disappeared")
                    db.commit()
                    return _inbox_message_from_row(linked)

                expected_key = f"assigned-worker-completion:{callback.completion_id}"
                existing_server = (
                    db.query(InboxModel).filter(InboxModel.idempotency_key == expected_key).first()
                )
                if existing_server is not None:
                    _validate_server_completion_inbox_row(callback, existing_server)
                    _link_callback(callback, existing_server)
                    db.flush()
                    _validate_assigned_worker_callback_row(db, callback)
                    db.commit()
                    return _inbox_message_from_row(existing_server)

                explicit_row = _new_row(
                    row_origin=InboxMessageOrigin.EXPLICIT,
                    row_assignment_id=callback.assignment_id,
                    row_idempotency_key=None,
                )
                _link_callback(callback, explicit_row)
                db.flush()
                _validate_assigned_worker_callback_row(db, callback)
                db.commit()
                return _inbox_message_from_row(explicit_row)

        if origin == InboxMessageOrigin.SERVER_COMPLETION:
            if callback is None:
                raise ValueError("Server completion requires a persisted assignment")
            if callback.lifecycle != AssignmentLifecycle.COMPLETED.value:
                raise ValueError("Server completion requires a captured successful result")
            if callback.delivery_state == CompletionDeliveryState.TERMINAL_ERROR.value:
                raise ValueError("A terminal callback cannot be enqueued again")
            expected_message = format_server_completion_message(
                callback.final_result or "",
                callback.worker_terminal_id,
                callback.assignment_id,
                callback.completion_id,
            )
            expected_key = f"assigned-worker-completion:{callback.completion_id}"
            if message != expected_message or idempotency_key != expected_key:
                raise ValueError("Server completion payload or idempotency identity mismatch")

            if callback.inbox_message_id is not None:
                linked = (
                    db.query(InboxModel).filter(InboxModel.id == callback.inbox_message_id).first()
                )
                if linked is None:
                    raise _callback_integrity_error(callback, "linked inbox row disappeared")
                db.commit()
                return _inbox_message_from_row(linked)

            existing_server = (
                db.query(InboxModel).filter(InboxModel.idempotency_key == expected_key).first()
            )
            if existing_server is not None:
                _validate_server_completion_inbox_row(callback, existing_server)
                _link_callback(callback, existing_server)
                db.flush()
                _validate_assigned_worker_callback_row(db, callback)
                db.commit()
                return _inbox_message_from_row(existing_server)

            explicit_candidates = (
                db.query(InboxModel)
                .filter(
                    InboxModel.assignment_id == callback.assignment_id,
                    InboxModel.origin == InboxMessageOrigin.EXPLICIT.value,
                    InboxModel.status.in_(
                        (
                            MessageStatus.PENDING.value,
                            MessageStatus.DELIVERING.value,
                            MessageStatus.DELIVERED.value,
                        )
                    ),
                )
                .order_by(InboxModel.created_at.asc(), InboxModel.id.asc())
                .all()
            )
            for explicit_row in explicit_candidates:
                if canonical_callback_text(explicit_row.message) == canonical_callback_text(
                    callback.final_result or ""
                ):
                    _link_callback(callback, explicit_row)
                    db.flush()
                    _validate_assigned_worker_callback_row(db, callback)
                    db.commit()
                    return _inbox_message_from_row(explicit_row)

            server_row = _new_row(
                row_origin=InboxMessageOrigin.SERVER_COMPLETION,
                row_assignment_id=callback.assignment_id,
                row_idempotency_key=expected_key,
            )
            _link_callback(callback, server_row)
            db.flush()
            _validate_assigned_worker_callback_row(db, callback)
            db.commit()
            return _inbox_message_from_row(server_row)

        if idempotency_key:
            existing = (
                db.query(InboxModel).filter(InboxModel.idempotency_key == idempotency_key).first()
            )
            if existing is not None:
                if not _matches_immutable_payload(existing):
                    raise ValueError(f"Inbox idempotency key collision: {idempotency_key}")
                db.commit()
                return _inbox_message_from_row(existing)

        inbox_msg = _new_row(row_assignment_id=assignment_id)
        db.commit()
        return _inbox_message_from_row(inbox_msg)


def get_pending_messages(receiver_id: str, limit: int = 1) -> List[InboxMessage]:
    """Get pending messages ordered by created_at ASC (oldest first)."""
    return get_inbox_messages(receiver_id, limit=limit, status=MessageStatus.PENDING)


def claim_inbox_message(message_id: int, claim_token: str) -> Optional[InboxMessage]:
    """Atomically claim one PENDING row; only one concurrent caller can win."""
    if not claim_token:
        raise ValueError("claim_token must be non-empty")
    with SessionLocal() as db:
        claimed_at = datetime.now()
        claimed = (
            db.query(InboxModel)
            .filter(
                InboxModel.id == message_id,
                InboxModel.status == MessageStatus.PENDING.value,
            )
            .update(
                {
                    InboxModel.status: MessageStatus.DELIVERING.value,
                    InboxModel.claim_token: claim_token,
                    InboxModel.claimed_at: claimed_at,
                },
                synchronize_session=False,
            )
        )
        if claimed != 1:
            db.rollback()
            return None
        row = db.query(InboxModel).filter(InboxModel.id == message_id).first()
        if row is None:  # Defensive: the row cannot disappear inside this write transaction.
            db.rollback()
            return None
        _validate_inbox_callback_evidence(db, row)
        db.commit()
        return _inbox_message_from_row(row)


def resolve_inbox_claim(
    message_id: int,
    claim_token: str,
    status: MessageStatus,
) -> bool:
    """Resolve only the exact durable claim owned by ``claim_token``."""
    if status not in (
        MessageStatus.PENDING,
        MessageStatus.DELIVERED,
        MessageStatus.FAILED,
    ):
        raise ValueError(f"Invalid inbox claim resolution status: {status.value}")
    with SessionLocal() as db:
        # Serialize claim resolution with receiver deletion. Whichever commits
        # first supplies the durable truth: DELIVERED means paste finished
        # before deletion; FAILED/TERMINAL_ERROR means deletion won first.
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        values: Dict[Any, Any] = {
            InboxModel.status: status.value,
            InboxModel.claim_token: None,
        }
        updated = (
            db.query(InboxModel)
            .filter(
                InboxModel.id == message_id,
                InboxModel.status == MessageStatus.DELIVERING.value,
                InboxModel.claim_token == claim_token,
            )
            .update(values, synchronize_session=False)
        )
        if updated == 1:
            row = db.query(InboxModel).filter(InboxModel.id == message_id).first()
            if row is None:  # Defensive: this write transaction owns the row.
                db.rollback()
                return False
            _validate_inbox_callback_evidence(db, row)
        db.commit()
        return updated == 1


def is_assigned_worker_callback_inbox_message(message_id: int) -> bool:
    """Return whether an inbox row is the callback evidence linked by a worker."""
    with SessionLocal() as db:
        callback = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.inbox_message_id == message_id)
            .first()
        )
        if callback is None:
            return False
        _validate_assigned_worker_callback_row(db, callback)
        return True


def get_inbox_messages(
    receiver_id: str, limit: int = 10, status: Optional[MessageStatus] = None
) -> List[InboxMessage]:
    """Get inbox messages with optional status filter ordered by created_at ASC (oldest first).

    Args:
        receiver_id: Terminal ID to get messages for
        limit: Maximum number of messages to return (default: 10)
        status: Optional filter by message status (None = all statuses)

    Returns:
        List of inbox messages ordered by creation time (oldest first)
    """
    with SessionLocal() as db:
        query = db.query(InboxModel).filter(InboxModel.receiver_id == receiver_id)

        if status is not None:
            query = query.filter(InboxModel.status == status.value)

        messages = query.order_by(InboxModel.created_at.asc()).limit(limit).all()
        for message in messages:
            _validate_inbox_callback_evidence(db, message)

        return [_inbox_message_from_row(msg) for msg in messages]


def _callback_integrity_error(
    row: AssignedWorkerCallbackModel, detail: str
) -> AssignedWorkerIntegrityError:
    return AssignedWorkerIntegrityError(
        f"Assigned-worker callback integrity failure for assignment "
        f"{row.assignment_id!r}: {detail}"
    )


def _validate_assigned_worker_callback_row(db: Any, row: AssignedWorkerCallbackModel) -> None:
    """Fail closed on corrupt route, result, state, timestamps, or inbox evidence."""
    try:
        lifecycle = AssignmentLifecycle(row.lifecycle)
        delivery_state = CompletionDeliveryState(row.delivery_state)
        CompletionReceiverState(row.receiver_state)
    except ValueError as exc:
        raise _callback_integrity_error(row, f"unknown enum value: {exc}") from exc

    expected_routing_digest = callback_routing_digest(
        row.assignment_id,
        row.completion_id,
        row.worker_terminal_id,
        row.caller_id,
    )
    if row.routing_digest != expected_routing_digest:
        raise _callback_integrity_error(row, "immutable routing digest mismatch")

    result_fields = (row.final_result, row.final_result_sha256, row.result_reference)
    if any(value is None for value in result_fields) != all(
        value is None for value in result_fields
    ):
        raise _callback_integrity_error(row, "captured result fields are only partially present")
    has_result = row.final_result is not None
    if has_result:
        expected_hash = hashlib.sha256(row.final_result.encode("utf-8")).hexdigest()
        if row.final_result_sha256 != expected_hash:
            raise _callback_integrity_error(row, "final_result SHA-256 mismatch")
        expected_reference = f"assigned-worker-callback:{row.assignment_id}"
        if row.result_reference != expected_reference:
            raise _callback_integrity_error(row, "result_reference does not match assignment")

    attempt_count = row.attempt_count or 0
    if attempt_count < 0:
        raise _callback_integrity_error(row, "attempt_count is negative")
    if attempt_count == 0 and (row.first_attempt_at is not None or row.last_attempt_at is not None):
        raise _callback_integrity_error(row, "attempt timestamps exist with zero attempts")
    if attempt_count > 0 and (row.first_attempt_at is None or row.last_attempt_at is None):
        raise _callback_integrity_error(row, "attempt timestamps are missing")
    if (
        row.first_attempt_at is not None
        and row.last_attempt_at is not None
        and row.last_attempt_at < row.first_attempt_at
    ):
        raise _callback_integrity_error(row, "last_attempt_at precedes first_attempt_at")

    ordered_timestamps = (
        ("dispatched_at", row.dispatched_at, row.created_at),
        ("completion_observed_at", row.completion_observed_at, row.dispatched_at),
        ("captured_at", row.captured_at, row.completion_observed_at),
        ("last_attempt_at", row.last_attempt_at, row.first_attempt_at),
        ("acknowledged_at", row.acknowledged_at, row.enqueued_at),
        ("terminal_error_at", row.terminal_error_at, row.created_at),
    )
    for timestamp_name, later, earlier in ordered_timestamps:
        if later is not None and earlier is not None and later < earlier:
            raise _callback_integrity_error(
                row, f"{timestamp_name} precedes its required predecessor"
            )

    if lifecycle == AssignmentLifecycle.ASSIGNED:
        if row.dispatched_at is not None or has_result:
            raise _callback_integrity_error(
                row, "unproven assignment contains dispatch/result data"
            )
        if delivery_state != CompletionDeliveryState.NOT_READY:
            raise _callback_integrity_error(
                row, "unproven assignment has an impossible delivery state"
            )
        if any(
            value is not None
            for value in (
                row.completion_observed_at,
                row.captured_at,
                row.first_attempt_at,
                row.last_attempt_at,
                row.enqueued_at,
                row.acknowledged_at,
                row.terminal_error_at,
                row.inbox_message_id,
            )
        ):
            raise _callback_integrity_error(row, "unproven assignment contains terminal evidence")
    elif lifecycle == AssignmentLifecycle.DISPATCHED:
        if row.dispatched_at is None or has_result:
            raise _callback_integrity_error(
                row, "dispatched assignment has invalid dispatch/result data"
            )
        if delivery_state not in (
            CompletionDeliveryState.NOT_READY,
            CompletionDeliveryState.RETRYABLE,
        ):
            raise _callback_integrity_error(
                row, "dispatched assignment has an impossible delivery state"
            )
        if any(
            value is not None
            for value in (
                row.completion_observed_at,
                row.captured_at,
                row.enqueued_at,
                row.acknowledged_at,
                row.terminal_error_at,
                row.inbox_message_id,
            )
        ):
            raise _callback_integrity_error(row, "dispatched assignment contains terminal evidence")
    elif lifecycle == AssignmentLifecycle.UNRESOLVED:
        if has_result or delivery_state != CompletionDeliveryState.MANUAL_RECOVERY:
            raise _callback_integrity_error(
                row, "unresolved assignment has an impossible result/state"
            )
        if row.terminal_error_at is None or not row.last_error:
            raise _callback_integrity_error(row, "unresolved assignment lacks recovery audit data")
        if any(
            value is not None
            for value in (
                row.completion_observed_at,
                row.captured_at,
                row.first_attempt_at,
                row.last_attempt_at,
                row.enqueued_at,
                row.acknowledged_at,
                row.inbox_message_id,
            )
        ):
            raise _callback_integrity_error(row, "unresolved assignment contains delivery evidence")
    elif lifecycle == AssignmentLifecycle.COMPLETED:
        if (
            row.dispatched_at is None
            or row.completion_observed_at is None
            or row.captured_at is None
            or not has_result
        ):
            raise _callback_integrity_error(row, "completed assignment lacks capture evidence")
        if delivery_state in (
            CompletionDeliveryState.NOT_READY,
            CompletionDeliveryState.MANUAL_RECOVERY,
        ):
            raise _callback_integrity_error(
                row, "completed assignment has an impossible delivery state"
            )
        if row.first_attempt_at is not None and row.first_attempt_at < row.captured_at:
            raise _callback_integrity_error(row, "delivery attempt precedes result capture")
        if row.acknowledged_at is not None and row.acknowledged_at < row.captured_at:
            raise _callback_integrity_error(row, "callback acknowledgement precedes result capture")
        if row.terminal_error_at is not None and row.terminal_error_at < row.captured_at:
            raise _callback_integrity_error(row, "terminal error precedes result capture")
    elif lifecycle in (AssignmentLifecycle.FAILED, AssignmentLifecycle.CANCELLED):
        if has_result or delivery_state != CompletionDeliveryState.TERMINAL_ERROR:
            raise _callback_integrity_error(
                row, "failed/cancelled assignment has invalid result/state"
            )
        if any(
            value is not None
            for value in (
                row.completion_observed_at,
                row.captured_at,
                row.first_attempt_at,
                row.last_attempt_at,
                row.enqueued_at,
                row.acknowledged_at,
                row.inbox_message_id,
            )
        ):
            raise _callback_integrity_error(
                row, "failed/cancelled assignment has delivery evidence"
            )

    if delivery_state == CompletionDeliveryState.NOT_READY and (
        attempt_count != 0 or row.inbox_message_id is not None
    ):
        raise _callback_integrity_error(row, "not-ready state contains delivery evidence")
    if delivery_state == CompletionDeliveryState.CAPTURED and (
        attempt_count != 0
        or row.inbox_message_id is not None
        or row.enqueued_at is not None
        or row.acknowledged_at is not None
    ):
        raise _callback_integrity_error(row, "captured state contains premature delivery evidence")
    if delivery_state == CompletionDeliveryState.DELIVERING and (
        attempt_count == 0
        or row.inbox_message_id is not None
        or row.enqueued_at is not None
        or row.acknowledged_at is not None
    ):
        raise _callback_integrity_error(row, "delivering state has an impossible evidence shape")
    if delivery_state == CompletionDeliveryState.RETRYABLE and (
        row.inbox_message_id is not None
        or row.enqueued_at is not None
        or row.acknowledged_at is not None
        or not row.last_error
    ):
        raise _callback_integrity_error(row, "retryable state has an impossible evidence shape")

    if delivery_state in (
        CompletionDeliveryState.ENQUEUED,
        CompletionDeliveryState.ACKNOWLEDGED,
        CompletionDeliveryState.SUPPRESSED_EXPLICIT,
    ) and (row.inbox_message_id is None or row.enqueued_at is None):
        raise _callback_integrity_error(row, "delivery state lacks linked inbox evidence")
    if (
        delivery_state
        in (
            CompletionDeliveryState.ACKNOWLEDGED,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT,
        )
        and row.acknowledged_at is None
    ):
        raise _callback_integrity_error(row, "acknowledged state lacks acknowledged_at")
    if delivery_state == CompletionDeliveryState.ENQUEUED and row.acknowledged_at is not None:
        raise _callback_integrity_error(row, "enqueued state has premature acknowledgement")
    if delivery_state == CompletionDeliveryState.TERMINAL_ERROR and (
        row.terminal_error_at is None or not row.last_error
    ):
        raise _callback_integrity_error(row, "terminal error lacks error audit data")
    if delivery_state == CompletionDeliveryState.MANUAL_RECOVERY and (
        row.terminal_error_at is None or not row.last_error
    ):
        raise _callback_integrity_error(row, "manual recovery lacks audit data")
    if row.inbox_message_id is None and (
        row.enqueued_at is not None or row.acknowledged_at is not None
    ):
        raise _callback_integrity_error(row, "inbox timestamps exist without an inbox link")

    if row.inbox_message_id is not None:
        inbox_row = db.query(InboxModel).filter(InboxModel.id == row.inbox_message_id).first()
        if inbox_row is None:
            raise _callback_integrity_error(row, "linked inbox evidence is missing")
        if (
            inbox_row.assignment_id != row.assignment_id
            or inbox_row.sender_id != row.worker_terminal_id
            or inbox_row.receiver_id != row.caller_id
        ):
            raise _callback_integrity_error(row, "linked inbox route/assignment mismatch")
        try:
            inbox_status = MessageStatus(inbox_row.status)
        except ValueError as exc:
            raise _callback_integrity_error(row, f"linked inbox status is invalid: {exc}") from exc
        if inbox_status == MessageStatus.DELIVERING:
            if not inbox_row.claim_token or inbox_row.claimed_at is None:
                raise _callback_integrity_error(row, "linked inbox delivery claim is incomplete")
        elif inbox_row.claim_token is not None:
            raise _callback_integrity_error(
                row, "linked inbox has a claim token outside DELIVERING"
            )
        if (
            delivery_state
            in (
                CompletionDeliveryState.ACKNOWLEDGED,
                CompletionDeliveryState.SUPPRESSED_EXPLICIT,
            )
            and inbox_status == MessageStatus.FAILED
        ):
            raise _callback_integrity_error(row, "successful enqueue evidence is marked failed")
        if (
            delivery_state == CompletionDeliveryState.TERMINAL_ERROR
            and lifecycle == AssignmentLifecycle.COMPLETED
            and inbox_status != MessageStatus.FAILED
        ):
            raise _callback_integrity_error(
                row, "undeliverable linked callback is not marked failed"
            )
        if inbox_row.origin == InboxMessageOrigin.SERVER_COMPLETION.value:
            expected_key = f"assigned-worker-completion:{row.completion_id}"
            expected_message = format_server_completion_message(
                row.final_result or "",
                row.worker_terminal_id,
                row.assignment_id,
                row.completion_id,
            )
            if inbox_row.idempotency_key != expected_key or inbox_row.message != expected_message:
                raise _callback_integrity_error(row, "server inbox payload/idempotency mismatch")
            if row.enqueued_at is not None and row.enqueued_at < row.captured_at:
                raise _callback_integrity_error(row, "server enqueue precedes result capture")
            if delivery_state == CompletionDeliveryState.SUPPRESSED_EXPLICIT:
                raise _callback_integrity_error(row, "suppressed state links a server callback")
        elif inbox_row.origin == InboxMessageOrigin.EXPLICIT.value:
            if inbox_row.idempotency_key is not None:
                raise _callback_integrity_error(
                    row, "explicit callback has a server idempotency key"
                )
            if not has_result or canonical_callback_text(
                inbox_row.message
            ) != canonical_callback_text(row.final_result or ""):
                raise _callback_integrity_error(
                    row, "explicit inbox evidence is not final-equivalent"
                )
            if delivery_state not in (
                CompletionDeliveryState.SUPPRESSED_EXPLICIT,
                CompletionDeliveryState.TERMINAL_ERROR,
            ):
                raise _callback_integrity_error(row, "non-suppressed state links explicit evidence")
        else:
            raise _callback_integrity_error(row, "linked inbox origin is not callback evidence")


def _validate_inbox_callback_evidence(db: Any, inbox_row: InboxModel) -> None:
    """Prevent a tampered callback row from reaching terminal paste/readback."""
    callback = (
        db.query(AssignedWorkerCallbackModel)
        .filter(AssignedWorkerCallbackModel.inbox_message_id == inbox_row.id)
        .first()
    )
    if callback is None and inbox_row.origin == InboxMessageOrigin.SERVER_COMPLETION.value:
        callback = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == inbox_row.assignment_id)
            .first()
        )
        if callback is None:
            raise AssignedWorkerIntegrityError(
                f"Server-completion inbox row {inbox_row.id} has no persisted assignment"
            )
        _validate_assigned_worker_callback_row(db, callback)
        _validate_server_completion_inbox_row(callback, inbox_row)
        return
    if callback is not None:
        _validate_assigned_worker_callback_row(db, callback)


def _commit_validated_callback_mutation(db: Any, row: AssignedWorkerCallbackModel) -> None:
    """Validate pending callback state before, and again after, committing it.

    Readback-only validation is too late: a malformed transition would already
    be durable. Flushing first lets SQLite triggers and the complete state/link
    validator reject the mutation while rollback can still restore the row.
    """
    db.flush()
    _validate_assigned_worker_callback_row(db, row)
    db.commit()
    db.refresh(row)
    _validate_assigned_worker_callback_row(db, row)


def _assigned_worker_callback_from_row(
    row: AssignedWorkerCallbackModel,
) -> AssignedWorkerCallback:
    """Convert an already integrity-validated callback row to the read model."""
    return AssignedWorkerCallback(
        assignment_id=row.assignment_id,
        completion_id=row.completion_id,
        worker_terminal_id=row.worker_terminal_id,
        caller_id=row.caller_id,
        routing_digest=row.routing_digest,
        lifecycle=AssignmentLifecycle(row.lifecycle),
        delivery_state=CompletionDeliveryState(row.delivery_state),
        receiver_state=CompletionReceiverState(row.receiver_state),
        final_result=row.final_result,
        final_result_sha256=row.final_result_sha256,
        result_reference=row.result_reference,
        inbox_message_id=row.inbox_message_id,
        attempt_count=row.attempt_count or 0,
        created_at=row.created_at,
        dispatched_at=row.dispatched_at,
        completion_observed_at=row.completion_observed_at,
        captured_at=row.captured_at,
        first_attempt_at=row.first_attempt_at,
        last_attempt_at=row.last_attempt_at,
        enqueued_at=row.enqueued_at,
        acknowledged_at=row.acknowledged_at,
        terminal_error_at=row.terminal_error_at,
        last_error=row.last_error,
    )


def get_assigned_worker_callback(worker_terminal_id: str) -> Optional[AssignedWorkerCallback]:
    """Return the durable callback record for a worker, even after retirement."""
    with SessionLocal() as db:
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.worker_terminal_id == worker_terminal_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        return _assigned_worker_callback_from_row(row)


def get_assigned_worker_callback_by_assignment(
    assignment_id: str,
) -> Optional[AssignedWorkerCallback]:
    """Return one callback record by immutable assignment identity."""
    with SessionLocal() as db:
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        return _assigned_worker_callback_from_row(row)


def list_reconcilable_assigned_worker_callbacks() -> List[AssignedWorkerCallback]:
    """List callbacks that may still require status capture or durable enqueue."""
    terminal_states = (
        CompletionDeliveryState.ACKNOWLEDGED.value,
        CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
        CompletionDeliveryState.MANUAL_RECOVERY.value,
        CompletionDeliveryState.TERMINAL_ERROR.value,
    )
    with SessionLocal() as db:
        rows = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.delivery_state.notin_(terminal_states))
            .order_by(AssignedWorkerCallbackModel.created_at.asc())
            .all()
        )
        for row in rows:
            _validate_assigned_worker_callback_row(db, row)
        return [_assigned_worker_callback_from_row(row) for row in rows]


def list_protected_assigned_worker_callbacks() -> List[AssignedWorkerCallback]:
    """List workers whose uncaptured terminal remains a recovery handle."""
    with SessionLocal() as db:
        rows = (
            db.query(AssignedWorkerCallbackModel)
            .filter(
                AssignedWorkerCallbackModel.lifecycle.in_(
                    (
                        AssignmentLifecycle.ASSIGNED.value,
                        AssignmentLifecycle.DISPATCHED.value,
                        AssignmentLifecycle.UNRESOLVED.value,
                    )
                )
            )
            .order_by(AssignedWorkerCallbackModel.created_at.asc())
            .all()
        )
        for row in rows:
            _validate_assigned_worker_callback_row(db, row)
        return [_assigned_worker_callback_from_row(row) for row in rows]


def mark_assigned_worker_dispatched(worker_terminal_id: str) -> Optional[AssignedWorkerCallback]:
    """Persist that the assigned prompt was accepted by the worker."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.worker_terminal_id == worker_terminal_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.lifecycle == AssignmentLifecycle.ASSIGNED.value:
            row.lifecycle = AssignmentLifecycle.DISPATCHED.value
            row.dispatched_at = datetime.now()
            _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def capture_assigned_worker_completion(
    worker_terminal_id: str,
    final_result: str,
    final_result_sha256: str,
    result_reference: str,
) -> Optional[AssignedWorkerCallback]:
    """Capture a final report once, after a dispatched worker truly completes."""
    calculated_sha256 = hashlib.sha256(final_result.encode("utf-8")).hexdigest()
    if final_result_sha256 != calculated_sha256:
        raise ValueError("final_result_sha256 does not match final_result")
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.worker_terminal_id == worker_terminal_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.lifecycle == AssignmentLifecycle.COMPLETED.value:
            if row.final_result_sha256 != final_result_sha256:
                logger.warning(
                    "Ignoring changed final output for immutable completion %s",
                    row.completion_id,
                )
            return _assigned_worker_callback_from_row(row)
        if row.lifecycle != AssignmentLifecycle.DISPATCHED.value:
            return _assigned_worker_callback_from_row(row)
        expected_reference = f"assigned-worker-callback:{row.assignment_id}"
        if result_reference != expected_reference:
            raise ValueError(
                f"result_reference must be the immutable assignment reference {expected_reference!r}"
            )

        now = datetime.now()
        row.lifecycle = AssignmentLifecycle.COMPLETED.value
        row.delivery_state = CompletionDeliveryState.CAPTURED.value
        row.completion_observed_at = now
        row.captured_at = now
        row.final_result = final_result
        row.final_result_sha256 = final_result_sha256
        row.result_reference = result_reference
        row.last_error = None
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def list_explicit_callback_candidates(assignment_id: str) -> List[InboxMessage]:
    """Return durable, non-failed explicit messages associated with an assignment."""
    with SessionLocal() as db:
        callback = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if callback is None:
            return []
        _validate_assigned_worker_callback_row(db, callback)
        rows = (
            db.query(InboxModel)
            .filter(
                InboxModel.assignment_id == assignment_id,
                InboxModel.origin == InboxMessageOrigin.EXPLICIT.value,
                InboxModel.status.in_(
                    (
                        MessageStatus.PENDING.value,
                        MessageStatus.DELIVERING.value,
                        MessageStatus.DELIVERED.value,
                    )
                ),
            )
            .order_by(InboxModel.created_at.asc(), InboxModel.id.asc())
            .all()
        )
        for row in rows:
            if (
                row.sender_id != callback.worker_terminal_id
                or row.receiver_id != callback.caller_id
            ):
                raise _callback_integrity_error(
                    callback, f"explicit inbox candidate {row.id} has a routing mismatch"
                )
        return [_inbox_message_from_row(row) for row in rows]


def record_completion_delivery_attempt(
    assignment_id: str, receiver_state: CompletionReceiverState
) -> Optional[AssignedWorkerCallback]:
    """Durably record an enqueue attempt before touching the inbox table."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ENQUEUED.value,
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        now = datetime.now()
        row.delivery_state = CompletionDeliveryState.DELIVERING.value
        row.receiver_state = receiver_state.value
        row.attempt_count = (row.attempt_count or 0) + 1
        row.first_attempt_at = row.first_attempt_at or now
        row.last_attempt_at = now
        row.last_error = None
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def mark_completion_enqueued(
    assignment_id: str,
    inbox_message_id: int,
    receiver_state: CompletionReceiverState,
) -> Optional[AssignedWorkerCallback]:
    """Link the committed inbox row before acknowledging delivery."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        if row.inbox_message_id is not None and row.inbox_message_id != inbox_message_id:
            raise ValueError(
                f"Assignment {assignment_id} is already linked to inbox row "
                f"{row.inbox_message_id}"
            )
        inbox_row = db.query(InboxModel).filter(InboxModel.id == inbox_message_id).first()
        _validate_server_completion_inbox_row(row, inbox_row)
        row.delivery_state = CompletionDeliveryState.ENQUEUED.value
        row.receiver_state = receiver_state.value
        row.inbox_message_id = inbox_message_id
        row.enqueued_at = row.enqueued_at or datetime.now()
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def acknowledge_completion_enqueued(
    assignment_id: str, inbox_message_id: int
) -> Optional[AssignedWorkerCallback]:
    """Acknowledge only after the linked inbox row is durably observable."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        if row.delivery_state != CompletionDeliveryState.ENQUEUED.value:
            raise ValueError(f"Assignment {assignment_id} is not awaiting enqueue acknowledgement")
        if row.inbox_message_id != inbox_message_id:
            raise ValueError(
                f"Assignment {assignment_id} is linked to inbox row {row.inbox_message_id}, "
                f"not {inbox_message_id}"
            )
        inbox_row = db.query(InboxModel).filter(InboxModel.id == inbox_message_id).first()
        _validate_server_completion_inbox_row(row, inbox_row)
        row.delivery_state = CompletionDeliveryState.ACKNOWLEDGED.value
        row.inbox_message_id = inbox_message_id
        row.acknowledged_at = datetime.now()
        row.last_error = None
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def _validate_server_completion_inbox_row(
    callback_row: AssignedWorkerCallbackModel,
    inbox_row: Optional[InboxModel],
) -> None:
    """Require the exact immutable callback route and idempotency identity."""
    expected_key = f"assigned-worker-completion:{callback_row.completion_id}"
    if (
        inbox_row is None
        or inbox_row.assignment_id != callback_row.assignment_id
        or inbox_row.sender_id != callback_row.worker_terminal_id
        or inbox_row.receiver_id != callback_row.caller_id
        or inbox_row.origin != InboxMessageOrigin.SERVER_COMPLETION.value
        or inbox_row.idempotency_key != expected_key
        or inbox_row.message
        != format_server_completion_message(
            callback_row.final_result or "",
            callback_row.worker_terminal_id,
            callback_row.assignment_id,
            callback_row.completion_id,
        )
    ):
        raise ValueError(
            f"Inbox row is not the immutable server completion for assignment "
            f"{callback_row.assignment_id}"
        )


def suppress_completion_for_explicit_callback(
    assignment_id: str,
    inbox_message_id: int,
    receiver_state: CompletionReceiverState,
) -> Optional[AssignedWorkerCallback]:
    """Record that an equivalent explicit final callback already exists."""
    with SessionLocal() as db:
        # This legacy recovery API obeys the same writer serialization as the
        # explicit send endpoint and server callback insertion.
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        inbox_row = db.query(InboxModel).filter(InboxModel.id == inbox_message_id).first()
        if (
            row.lifecycle != AssignmentLifecycle.COMPLETED.value
            or row.final_result is None
            or inbox_row is None
            or inbox_row.assignment_id != assignment_id
            or inbox_row.sender_id != row.worker_terminal_id
            or inbox_row.receiver_id != row.caller_id
            or inbox_row.origin != InboxMessageOrigin.EXPLICIT.value
            or inbox_row.idempotency_key is not None
            or inbox_row.status
            not in (
                MessageStatus.PENDING.value,
                MessageStatus.DELIVERING.value,
                MessageStatus.DELIVERED.value,
            )
            or canonical_callback_text(inbox_row.message)
            != canonical_callback_text(row.final_result)
        ):
            raise ValueError(
                "Explicit callback suppression requires an equivalent routed final inbox row"
            )
        if row.inbox_message_id is not None and row.inbox_message_id != inbox_message_id:
            raise ValueError(
                f"Assignment {assignment_id} is already linked to inbox row "
                f"{row.inbox_message_id}"
            )
        now = datetime.now()
        row.delivery_state = CompletionDeliveryState.SUPPRESSED_EXPLICIT.value
        row.receiver_state = receiver_state.value
        row.inbox_message_id = inbox_message_id
        row.enqueued_at = row.enqueued_at or inbox_row.created_at or now
        row.acknowledged_at = now
        row.last_error = None
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def mark_completion_retryable(
    assignment_id: str,
    error: str,
    receiver_state: CompletionReceiverState = CompletionReceiverState.RETRYABLE_FAILURE,
) -> Optional[AssignedWorkerCallback]:
    """Retain a transient delivery error for server-side reconciliation."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        row.delivery_state = CompletionDeliveryState.RETRYABLE.value
        row.receiver_state = receiver_state.value
        row.last_error = error
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def mark_completion_terminal_error(
    assignment_id: str,
    error: str,
    receiver_state: CompletionReceiverState,
    *,
    lifecycle: Optional[AssignmentLifecycle] = None,
) -> Optional[AssignedWorkerCallback]:
    """Set a permanent/manual-recovery terminal state without deleting report data."""
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.delivery_state in (
            CompletionDeliveryState.ACKNOWLEDGED.value,
            CompletionDeliveryState.SUPPRESSED_EXPLICIT.value,
            CompletionDeliveryState.TERMINAL_ERROR.value,
        ):
            return _assigned_worker_callback_from_row(row)
        if lifecycle is not None and row.lifecycle != AssignmentLifecycle.COMPLETED.value:
            row.lifecycle = lifecycle.value
        row.delivery_state = CompletionDeliveryState.TERMINAL_ERROR.value
        row.receiver_state = receiver_state.value
        row.terminal_error_at = datetime.now()
        row.last_error = error
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def mark_assignment_manual_recovery(
    assignment_id: str,
    error: str,
) -> Optional[AssignedWorkerCallback]:
    """Fail closed when restart status cannot prove task-input dispatch.

    The terminal is deliberately retained as the recovery handle.  This is a
    terminal audit state for automation, but unlike FAILED/CANCELLED it does not
    assert an outcome that CAO never observed.
    """
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = (
            db.query(AssignedWorkerCallbackModel)
            .filter(AssignedWorkerCallbackModel.assignment_id == assignment_id)
            .first()
        )
        if row is None:
            return None
        _validate_assigned_worker_callback_row(db, row)
        if row.lifecycle != AssignmentLifecycle.ASSIGNED.value:
            return _assigned_worker_callback_from_row(row)
        row.lifecycle = AssignmentLifecycle.UNRESOLVED.value
        row.delivery_state = CompletionDeliveryState.MANUAL_RECOVERY.value
        row.terminal_error_at = datetime.now()
        row.last_error = error
        _commit_validated_callback_mutation(db, row)
        return _assigned_worker_callback_from_row(row)


def record_project_alias(project_id: str, alias: str, kind: str) -> None:
    """Idempotently record a project_id ↔ alias mapping (Phase 2.5 U6).

    Used opportunistically by ``resolve_project_id`` to track historical
    cwd-hash and git-remote-url aliases for a canonical project_id. Best-effort
    only — DB errors are swallowed so identity resolution is never blocked.
    """
    if not project_id or not alias or project_id == alias:
        return
    try:
        with SessionLocal() as db:
            # Upsert by alias (the primary key). If the same alias was already
            # mapped — e.g. recorded against an override id, then re-resolved
            # via git remote — repoint it to the current canonical project_id
            # so reverse lookups stay deterministic instead of duplicating.
            existing = db.query(ProjectAliasModel).filter(ProjectAliasModel.alias == alias).first()
            if existing is None:
                db.add(ProjectAliasModel(project_id=project_id, alias=alias, kind=kind))
                db.commit()
            elif existing.project_id != project_id or existing.kind != kind:
                existing.project_id = project_id
                existing.kind = kind
                db.commit()
    except Exception as e:
        logger.debug(f"record_project_alias failed (non-fatal): {e}")


def get_project_id_by_alias(alias: str) -> Optional[str]:
    """Return the canonical ``project_id`` for an alias, or None if unknown."""
    if not alias:
        return None
    try:
        with SessionLocal() as db:
            row = db.query(ProjectAliasModel).filter(ProjectAliasModel.alias == alias).first()
            return cast(Optional[str], row.project_id) if row else None
    except Exception as e:
        logger.debug(f"get_project_id_by_alias failed (non-fatal): {e}")
        return None


def list_aliases_for_project(project_id: str) -> List[Dict[str, Any]]:
    """List all aliases recorded for a canonical ``project_id``."""
    if not project_id:
        return []
    try:
        with SessionLocal() as db:
            rows = (
                db.query(ProjectAliasModel).filter(ProjectAliasModel.project_id == project_id).all()
            )
            return [{"project_id": r.project_id, "alias": r.alias, "kind": r.kind} for r in rows]
    except Exception as e:
        logger.debug(f"list_aliases_for_project failed (non-fatal): {e}")
        return []


def update_message_status(message_id: int, status: MessageStatus) -> bool:
    """Update message status to MessageStatus.DELIVERED or MessageStatus.FAILED."""
    with SessionLocal() as db:
        message = db.query(InboxModel).filter(InboxModel.id == message_id).first()
        if message:
            message.status = status.value
            db.commit()
            return True
        return False


# Flow database functions


def create_flow(
    name: str,
    file_path: str,
    schedule: str,
    agent_profile: str,
    provider: str,
    script: str,
    next_run: datetime,
) -> Flow:
    """Create flow record."""
    with SessionLocal() as db:
        flow = FlowModel(
            name=name,
            file_path=file_path,
            schedule=schedule,
            agent_profile=agent_profile,
            provider=provider,
            script=script,
            next_run=next_run,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)
        return Flow(
            name=flow.name,
            file_path=flow.file_path,
            schedule=flow.schedule,
            agent_profile=flow.agent_profile,
            provider=flow.provider,
            script=flow.script,
            last_run=flow.last_run,
            next_run=flow.next_run,
            enabled=flow.enabled,
            prompt_template=None,
        )


def get_flow(name: str) -> Optional[Flow]:
    """Get flow by name."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if not flow:
            return None
        return Flow(
            name=flow.name,
            file_path=flow.file_path,
            schedule=flow.schedule,
            agent_profile=flow.agent_profile,
            provider=flow.provider,
            script=flow.script,
            last_run=flow.last_run,
            next_run=flow.next_run,
            enabled=flow.enabled,
            prompt_template=None,
        )


def list_flows() -> List[Flow]:
    """List all flows."""
    with SessionLocal() as db:
        flows = db.query(FlowModel).order_by(FlowModel.next_run).all()
        return [
            Flow(
                name=f.name,
                file_path=f.file_path,
                schedule=f.schedule,
                agent_profile=f.agent_profile,
                provider=f.provider,
                script=f.script,
                last_run=f.last_run,
                next_run=f.next_run,
                enabled=f.enabled,
                prompt_template=None,
            )
            for f in flows
        ]


def update_flow_run_times(name: str, last_run: datetime, next_run: datetime) -> bool:
    """Update flow run times after execution."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if flow:
            flow.last_run = last_run
            flow.next_run = next_run
            db.commit()
            return True
        return False


def update_flow_enabled(name: str, enabled: bool, next_run: Optional[datetime] = None) -> bool:
    """Update flow enabled status and optionally next_run."""
    with SessionLocal() as db:
        flow = db.query(FlowModel).filter(FlowModel.name == name).first()
        if flow:
            flow.enabled = enabled
            if next_run is not None:
                flow.next_run = next_run
            db.commit()
            return True
        return False


def delete_flow(name: str) -> bool:
    """Delete flow."""
    with SessionLocal() as db:
        deleted = db.query(FlowModel).filter(FlowModel.name == name).delete()
        db.commit()
        return deleted > 0


def get_flows_to_run() -> List[Flow]:
    """Get enabled flows where next_run <= now."""
    with SessionLocal() as db:
        now = datetime.now()
        flows = (
            db.query(FlowModel).filter(FlowModel.enabled == True, FlowModel.next_run <= now).all()
        )
        return [
            Flow(
                name=f.name,
                file_path=f.file_path,
                schedule=f.schedule,
                agent_profile=f.agent_profile,
                provider=f.provider,
                script=f.script,
                last_run=f.last_run,
                next_run=f.next_run,
                enabled=f.enabled,
                prompt_template=None,
            )
            for f in flows
        ]
