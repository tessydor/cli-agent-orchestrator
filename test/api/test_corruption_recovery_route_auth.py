"""Direct-REST authorization regression for the guarded native-dispatch
corruption capture-release route (message 1020).

The MCP wrapper test (test/mcp_server/test_release_corrupted_dispatch_
capture.py) proves the MCP tool cannot lie about its own identity -- but
that says nothing about a caller reaching the ROUTE directly, bypassing
the MCP tool entirely. This file exercises the route itself, over a real
FastAPI TestClient, with auth ENABLED and real scope enforcement (via
FastAPI's dependency-override mechanism, the same pattern
test_scope_coverage.py already uses), against a real, isolated,
file-backed SQLite database.

Findings this verifies (see the route's own docstring for the full
evidence trail):

- The auth layer (security/auth.py) grants only flat scopes
  (cao:read/cao:write/cao:admin) from the bearer token -- there is no
  subject/terminal-identity claim anywhere, so nothing at the transport
  layer ties a request to a specific terminal.
- requesting_caller_id is therefore a PLAIN BODY FIELD for a direct REST
  caller -- there is no cryptographic binding proving who supplied it.
- The route is deliberately ADMIN-only (not WRITE-or-ADMIN), matching the
  existing DELETE /terminals/{id} precedent (no ownership check, ADMIN-
  gated only).
- Even for an ADMIN-scoped caller, a requesting_caller_id that does not
  match the assignment's real recorded caller is refused by
  check_recovery_guards' GUARD_CALLER_MISMATCH -- a real, load-bearing
  backstop, not merely a scope check.

All content here is synthetic fixture data invented for this test --
nothing from any real assignment, terminal, or archive is copied into
this repository.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.models.inbox import InboxMessageOrigin, MessageStatus
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.security import auth
from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    assigned_worker_completion_service,
)
from cli_agent_orchestrator.services.native_dispatch_recovery import utf8_sha256

REAL_CALLER_ID = "11112222"
IMPOSTER_CALLER_ID = "99998888"
WORKER_ID = "cccc5555"
ASSIGNMENT_ID = "assignment-1020-rest"
COMPLETION_ID = "completion-1020-rest"

BOUND_PREFIX = "synthetic assigned task via the direct REST route (1020)"
MESSAGE_874_CONTENT = "synthetic unrelated follow-up (874 analogue) via the direct REST route"
CORRUPTED_TRANSCRIPT = BOUND_PREFIX + MESSAGE_874_CONTENT

ROUTE = f"/assigned-workers/{WORKER_ID}/corruption-recovery"


@pytest.fixture
def isolated_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'rest-corruption-recovery.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with patch("cli_agent_orchestrator.clients.database.SessionLocal", factory):
        yield engine


@pytest.fixture
def auth_on(monkeypatch):
    """Enable the auth layer for enforcement tests (same as test_scope_coverage.py)."""
    monkeypatch.setenv("CAO_AUTH_JWKS_URI", "https://idp.example/jwks")


@pytest.fixture(autouse=True)
def _clear_scope_override():
    yield
    app.dependency_overrides.pop(auth.get_current_scopes, None)


def _override_scopes(scopes):
    async def _dep():
        return list(scopes)

    return _dep


def _seed_incident() -> None:
    db.create_terminal(REAL_CALLER_ID, "cao-session", "developer-real", "mock_cli")
    db.create_terminal(
        WORKER_ID,
        "cao-session",
        "developer-worker",
        "mock_cli",
        caller_id=REAL_CALLER_ID,
        assignment_id=ASSIGNMENT_ID,
        completion_id=COMPLETION_ID,
    )
    dispatched = db.mark_assigned_worker_dispatched(WORKER_ID)
    assert dispatched is not None

    row_874 = db.create_inbox_message(
        REAL_CALLER_ID, WORKER_ID, MESSAGE_874_CONTENT, origin=InboxMessageOrigin.EXPLICIT
    )
    claimed = db.claim_inbox_message(row_874.id, "rest-claim-874")
    assert claimed is not None
    assert db.resolve_inbox_claim(row_874.id, "rest-claim-874", MessageStatus.DELIVERED)

    assigned_worker_completion_service.register_assignment(WORKER_ID)
    assigned_worker_completion_service.announce_terminal_status(WORKER_ID, TerminalStatus.COMPLETED)


def _body(**overrides) -> dict:
    body = dict(
        assignment_id=ASSIGNMENT_ID,
        requesting_caller_id=REAL_CALLER_ID,
        transcript_first_user_text=CORRUPTED_TRANSCRIPT,
        expected_transcript_sha256=utf8_sha256(CORRUPTED_TRANSCRIPT),
        admissible_dispatch_sha256=[utf8_sha256(BOUND_PREFIX)],
        concatenated_message_id="874",
        concatenated_message_sender_id=REAL_CALLER_ID,
        concatenated_message_receiver_id=WORKER_ID,
        concatenated_message_content=MESSAGE_874_CONTENT,
        concatenated_message_delivery_state="delivered",
        concatenated_message_session_id="synthetic-session",
        recorded_at="2026-09-14T00:00:00+00:00",
        acknowledgement=f"{REAL_CALLER_ID} acknowledges the corrupted dispatch is abandoned",
        acknowledged_at="2026-09-14T00:01:00+00:00",
    )
    body.update(overrides)
    return body


class TestCorruptionRecoveryRouteDirectRestAuth:
    def test_write_scope_alone_is_forbidden(self, client, isolated_db, auth_on):
        """A direct REST caller holding only cao:write is 403'd -- cannot
        even reach the guard, let alone supply a body-level identity claim.
        """
        try:
            _seed_incident()
            app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_WRITE])

            resp = client.post(ROUTE, json=_body())

            assert resp.status_code == 403, resp.text
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=0.2
                )
                is False
            ), "a 403 must never release the barrier"
        finally:
            assigned_worker_completion_service._release_capture_barrier(WORKER_ID)

    def test_admin_scope_spoofed_caller_is_still_refused_by_the_guard(
        self, client, isolated_db, auth_on
    ):
        """Message 1020's core finding, proven directly against the route:
        even an ADMIN-scoped direct REST caller supplying a
        requesting_caller_id that does NOT match the assignment's real
        recorded caller is refused by GUARD_CALLER_MISMATCH -- admin scope
        alone never manufactures a matching identity.
        """
        try:
            _seed_incident()
            app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_ADMIN])

            resp = client.post(ROUTE, json=_body(requesting_caller_id=IMPOSTER_CALLER_ID))

            assert resp.status_code == 409, resp.text
            assert "caller_mismatch" in resp.text
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=0.2
                )
                is False
            ), "a spoofed/mismatched caller must never release the barrier"
        finally:
            assigned_worker_completion_service._release_capture_barrier(WORKER_ID)

    def test_admin_scope_correct_caller_succeeds(self, client, isolated_db, auth_on):
        """The positive case: ADMIN scope plus a requesting_caller_id that
        DOES match the assignment's real recorded caller succeeds and
        releases the barrier -- the route is usable, not merely locked down.
        """
        try:
            _seed_incident()
            app.dependency_overrides[auth.get_current_scopes] = _override_scopes([auth.SCOPE_ADMIN])

            resp = client.post(ROUTE, json=_body())

            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert payload["success"] is True
            assert payload["released_now"] is True
            assert (
                assigned_worker_completion_service.wait_for_capture_before_input(
                    WORKER_ID, timeout=1.0
                )
                is True
            )
        finally:
            assigned_worker_completion_service._release_capture_barrier(WORKER_ID)
