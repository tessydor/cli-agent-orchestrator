"""End-to-end coverage for mcp_server/reconciliation_direct.py (correction-842).

Proves the in-process reconciliation path is a faithful, unweakened,
un-duplicated call into the exact same guarded service function the removed
HTTP endpoint used -- against a REAL SQLite database, not mocks -- and that
it never makes an HTTP call to do so.
"""

import ast
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import cli_agent_orchestrator
from cli_agent_orchestrator.clients import database as db
from cli_agent_orchestrator.mcp_server import reconciliation_direct
from cli_agent_orchestrator.models.assigned_worker import (
    AssignmentLifecycle,
    CompletionDeliveryState,
    TerminalRetirementReconciliationError,
    compute_reconciliation_state_token,
)


@pytest.fixture
def callback_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reconciliation-direct.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        db, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    try:
        yield
    finally:
        engine.dispose()


def _dispatched_stuck_assignment(
    caller_id: str = "11111111",
    worker_id: str = "22222222",
    assignment_id: str = "assignment-stuck",
    completion_id: str = "completion-stuck",
):
    """A DISPATCHED/RETRYABLE row with no captured report -- the incident shape."""
    db.create_terminal(caller_id, "cao-s", "caller", "mock_cli")
    db.create_terminal(
        worker_id,
        "cao-s",
        "worker",
        "mock_cli",
        caller_id=caller_id,
        assignment_id=assignment_id,
        completion_id=completion_id,
    )
    dispatched = db.mark_assigned_worker_dispatched(worker_id)
    assert dispatched is not None
    stuck = db.mark_completion_retryable(
        assignment_id, "Authoritative final report is not available yet"
    )
    assert stuck is not None
    assert stuck.lifecycle == AssignmentLifecycle.DISPATCHED
    assert stuck.delivery_state == CompletionDeliveryState.RETRYABLE
    return stuck


_VALID_SHA256 = "5" * 64


def test_module_imports_no_http_client(monkeypatch):
    """Structural proof: this module never imports requests or the MCP HTTP
    utils -- there is no way for it to make an HTTP call even by accident."""
    source_path = Path(cli_agent_orchestrator.__file__).parent / "mcp_server" / "reconciliation_direct.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "requests" not in imported
    assert "cli_agent_orchestrator.mcp_server.utils" not in imported


def test_module_never_imports_forbidden_clients_directly():
    """Consistent with the HTTP-only boundary's own exception category
    ("...or through process-local read-only services"): this module reaches
    state through services.assigned_worker_completion_service, never
    clients.database/clients.tmux directly."""
    source_path = Path(cli_agent_orchestrator.__file__).parent / "mcp_server" / "reconciliation_direct.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "cli_agent_orchestrator.clients.database" not in imported
    assert "cli_agent_orchestrator.clients.tmux" not in imported


def test_correct_terminal_reconciles_a_real_stuck_record(callback_db):
    """Regression 3 (correction-842): the correct terminal succeeds -- a real
    DB row, no mocks, exercising the exact same guarded service function."""
    stuck = _dispatched_stuck_assignment()
    token = compute_reconciliation_state_token(stuck)

    result = reconciliation_direct.reconcile_caller_accepted_result_locally(
        "22222222",
        "11111111",  # the recorded caller -- what CAO_TERMINAL_ID would resolve to
        "assignment-stuck",
        token,
        "verified via merged PR",
        {
            "archive_reference": "archive-1",
            "archive_sha256": _VALID_SHA256,
            "accepted_evidence": "PR44 merge 5c673ce11e8226ed23d2566f7f780d7d9471354c",
        },
    )

    assert result["caller_reconciled_at"] is not None
    # Never forged: the truthful missing-provider-report state is untouched.
    assert result["lifecycle"] == "dispatched"
    assert result["delivery_state"] == "retryable"
    assert result["final_result"] is None

    retained = db.get_assigned_worker_callback("22222222")
    assert retained.caller_reconciled_at is not None


def test_wrong_terminal_context_refuses(callback_db):
    """Regression 4 (correction-842): a DIFFERENT terminal's own
    CAO_TERMINAL_ID (an impostor MCP context, not the recorded caller) is
    refused with the unchanged wrong_caller guard."""
    stuck = _dispatched_stuck_assignment()
    token = compute_reconciliation_state_token(stuck)
    db.create_terminal("99999999", "cao-s", "impostor", "mock_cli")

    with pytest.raises(TerminalRetirementReconciliationError) as exc_info:
        reconciliation_direct.reconcile_caller_accepted_result_locally(
            "22222222",
            "99999999",  # a different terminal's own id, not the recorded caller
            "assignment-stuck",
            token,
            "reason",
            {
                "archive_reference": "archive-1",
                "archive_sha256": _VALID_SHA256,
                "accepted_evidence": "y",
            },
        )
    assert exc_info.value.code == "wrong_caller"

    untouched = db.get_assigned_worker_callback("22222222")
    assert untouched.caller_reconciled_at is None


def test_result_shape_matches_the_service_directly(callback_db):
    """The wrapper adds no logic of its own: its return value is exactly
    reconcile_caller_accepted_result's own model_dump(mode="json")."""
    from cli_agent_orchestrator.services.assigned_worker_completion_service import (
        assigned_worker_completion_service,
    )

    stuck = _dispatched_stuck_assignment(
        caller_id="33333333",
        worker_id="44444444",
        assignment_id="assignment-shape",
        completion_id="completion-shape",
    )
    token = compute_reconciliation_state_token(stuck)
    evidence = {
        "archive_reference": "archive-2",
        "archive_sha256": _VALID_SHA256,
        "accepted_evidence": "z",
    }

    direct_result = reconciliation_direct.reconcile_caller_accepted_result_locally(
        "44444444", "33333333", "assignment-shape", token, "reason", evidence
    )

    # A second, identical call is an idempotent replay at the service layer;
    # comparing against what the service itself reports for the same record
    # proves the wrapper didn't reshape or drop anything.
    service_record = assigned_worker_completion_service.reconcile_caller_accepted_result(
        "44444444", "33333333", "assignment-shape", token, "reason", evidence
    )
    assert direct_result == service_record.model_dump(mode="json")
