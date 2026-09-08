from unittest.mock import Mock

import pytest

from cli_agent_orchestrator.mcp_server import server


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("CAO_TERMINAL_ID", "11111111")
    get = Mock()
    monkeypatch.setattr(server.mcp_utils, "get_json", get)
    return get


def test_context_comes_from_server(api):
    api.side_effect = [{"id": "11111111", "caller_id": "22222222"}, {"working_directory": "/repo"}]
    r = server.get_terminal_context()
    assert r["success"] and r["terminal"]["caller_id"] == "22222222"
    assert r["approval"] == "routing_identity_only"


def test_context_mismatch_fails(api):
    api.return_value = {"id": "22222222"}
    assert not server.get_terminal_context()["success"]


def test_worker_output_is_bounded(api):
    api.side_effect = [
        {"id": "11111111", "session_name": "s"},
        {"id": "22222222", "caller_id": "11111111", "session_name": "s"},
        {"output": "x" * 30000},
    ]
    r = server.inspect_worker("22222222")
    assert r["success"] and len(r["output"]) == 24000 and r["truncated"]


@pytest.mark.parametrize("caller,session", [("33333333", "s"), ("11111111", "other")])
def test_unrelated_worker_output_not_read(api, caller, session):
    api.side_effect = [
        {"id": "11111111", "session_name": "s"},
        {"id": "22222222", "caller_id": caller, "session_name": session},
    ]
    assert not server.inspect_worker("22222222")["success"]
    assert api.call_count == 2
