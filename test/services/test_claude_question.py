from unittest.mock import MagicMock

import pytest

from cli_agent_orchestrator.services import claude_question as q
from cli_agent_orchestrator.services import terminal_service as t


def menu(selected=1, title="Choose implementation"):
    return (
        title
        + "\n"
        + "\n".join(
            f"{'❯' if i==selected else ' '} {i}. {label}"
            for i, label in enumerate(["Small change", "Broader change", "Type something."], 1)
        )
        + "\nEnter to select · ↑/↓ to navigate · Esc to cancel"
    )


@pytest.fixture
def transport(monkeypatch):
    q._PENDING.clear()
    q._CONSUMED.clear()
    state = {"selected": 1, "title": "Choose implementation", "keys": []}
    monkeypatch.setattr(q, "_screen", lambda *args: menu(state["selected"], state["title"]))

    def send(worker, key):
        state["keys"].append(key)
        if key == "Down":
            state["selected"] += 1
        if key == "Up":
            state["selected"] -= 1

    monkeypatch.setattr(t, "send_special_key", send)
    return state


def test_nondefault_once_no_paste(transport):
    s = q.snapshot("worker", "caller")
    q.answer("worker", "caller", s["prompt_sha256"], 2)
    assert transport["keys"] == ["Down", "Enter"]
    with pytest.raises(ValueError):
        q.answer("worker", "caller", s["prompt_sha256"], 2)
    assert transport["keys"] == ["Down", "Enter"]


@pytest.mark.parametrize("index", [0, 4, "2", True, 3])
def test_invalid_answer_sends_nothing(transport, index):
    s = q.snapshot("worker", "caller")
    with pytest.raises(ValueError):
        q.answer("worker", "caller", s["prompt_sha256"], index)
    assert transport["keys"] == []


def test_stale_menu_sends_nothing(transport):
    s = q.snapshot("worker", "caller")
    transport["title"] = "Different question"
    with pytest.raises(ValueError):
        q.answer("worker", "caller", s["prompt_sha256"], 2)
    assert transport["keys"] == []


def test_changed_during_navigation_never_sends_enter(transport, monkeypatch):
    s = q.snapshot("worker", "caller")

    def send(*args):
        transport["keys"].append(args[1])
        transport["title"] = "New question"

    monkeypatch.setattr(t, "send_special_key", send)
    with pytest.raises(ValueError):
        q.answer("worker", "caller", s["prompt_sha256"], 2)
    assert transport["keys"] == ["Down"]


def test_expired_sends_nothing(transport):
    s = q.snapshot("worker", "caller")
    q._PENDING[("worker", "caller")] = (s["prompt_sha256"], 0)
    with pytest.raises(ValueError):
        q.answer("worker", "caller", s["prompt_sha256"], 2)
    assert not transport["keys"]


def test_ambiguous_screen_rejected():
    with pytest.raises(ValueError):
        q.parse_screen(menu().replace("❯", " "))
    with pytest.raises(ValueError):
        q.parse_screen(menu() + "\nshell$\n" * 8)


def test_same_screen_cannot_be_rearmed_after_enter(transport):
    s = q.snapshot("worker", "caller")
    q.answer("worker", "caller", s["prompt_sha256"], 2)
    with pytest.raises(ValueError):
        q.snapshot("worker", "caller")
    assert transport["keys"] == ["Down", "Enter"]
