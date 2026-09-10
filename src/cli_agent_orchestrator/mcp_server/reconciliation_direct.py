"""Direct, in-process caller-evidence retirement reconciliation (correction-842).

The generic ``POST /assigned-workers/{id}/retirement-reconciliation`` HTTP
route was removed entirely (see the diff to ``api/main.py``): no caller
identity check reachable over HTTP can hold when the auth layer is disabled,
which is this deployment's actual running configuration (``is_auth_enabled()``
is False -- verified: no ``CAO_AUTH_LOCAL_TOKEN``, no ``AUTH0_DOMAIN``/
``CAO_AUTH_JWKS_URI`` configured). A ``require_any_scope``-style dependency,
or the local-service-token narrowing an earlier revision added, is a no-op in
that configuration, so the route stayed a fully unauthenticated, spoofable
mutation for the one deployment this fix is actually for.

This module therefore calls straight into
:meth:`AssignedWorkerCompletionService.reconcile_caller_accepted_result` --
the SAME guarded function the removed HTTP endpoint used, no logic duplicated
or weakened -- from *within this MCP server process*. The caller identity it
passes is derived exclusively from this process's own ``CAO_TERMINAL_ID`` (see
``mcp_server/server.py``'s ``_own_terminal_id_or_error``/
``utils.orchestration._current_terminal_id``), never a tool argument: there is
no way for a model or a generic HTTP client to make this call claim to be any
terminal other than the one this MCP server process actually is.

This stays fully within the existing HTTP-only MCP boundary
(``test/test_http_only_boundary.py``): the forbidden imports there are
``clients.tmux``/``clients.database`` directly, not the service layer built on
top of them -- the boundary test's own docstring already carves out "...or
through process-local read-only services" as a sanctioned alternative to the
HTTP surface, and ``mcp_server/server.py``'s ``memory_store``/``store_lesson``
tools already call straight into ``MemoryService`` (a genuine local WRITE,
not merely a read) for exactly this reason when no remote memory gateway is
configured. This module imports only ``services.assigned_worker_completion_service``
and ``models.assigned_worker`` -- never ``clients.database``/``clients.tmux``
-- so no exception to that test is needed; it follows the already-established
pattern rather than inventing a new one.

Two cross-process notes on why this is still correct:

- The service's in-memory per-worker lock (``AssignedWorkerCompletionService._worker_lock``)
  is process-local and therefore not shared with the real ``cao-server``
  process. That is not the correctness guarantee here: the SQLite
  ``BEGIN IMMEDIATE`` transaction plus the idempotent already-reconciled
  comparison inside ``mark_caller_reconciled``/``reconcile_caller_accepted_result``
  themselves DO serialize correctly across processes (the exact same
  guarantee every other write to this table already relies on). The
  in-process lock is redundant defense-in-depth within one process, not the
  source of correctness.
- Live-status detection (``AssignedWorkerCompletionService._detect_live_status``)
  reconstructs the provider on demand from durable terminal metadata
  (``providers.manager.ProviderManager.get_provider``'s own "create on-demand
  from database metadata" path) and reads tmux history directly -- the exact
  same "cold cache after a server restart" recovery path the real server
  itself already relies on. It does not depend on any in-memory registration
  this process never performed, so it derives a genuine, current status here
  too, not a stale/empty one.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from cli_agent_orchestrator.services.assigned_worker_completion_service import (
    assigned_worker_completion_service,
)


def reconcile_caller_accepted_result_locally(
    worker_terminal_id: str,
    caller_id: str,
    assignment_id: str,
    expected_state_token: str,
    reason: str,
    evidence: Mapping[str, str],
) -> Dict[str, Any]:
    """Call the service directly; never over HTTP.

    Raises the exact same ``TerminalRetirementReconciliationError`` (with its
    ``code``) the removed HTTP endpoint used to translate into a status code
    -- callers here (see ``mcp_server/server.py``) translate it into the same
    ``{"success": False, "error": "<code>: <message>"}`` shape every other
    MCP tool in this file already uses for a refused operation.
    """
    updated = assigned_worker_completion_service.reconcile_caller_accepted_result(
        worker_terminal_id,
        caller_id,
        assignment_id,
        expected_state_token,
        reason,
        evidence,
    )
    # mode="json": datetimes/enums become plain strings, matching what
    # jsonable_encoder(record.model_dump()) produced over the removed HTTP
    # path -- this dict is returned straight through the MCP transport, with
    # no FastAPI response layer to normalize it for us.
    return updated.model_dump(mode="json")
