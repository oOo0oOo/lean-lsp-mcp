from __future__ import annotations

import types

from lean_lsp_mcp import server


class FakeClient:
    def __init__(self) -> None:
        self.diagnostics_called = False
        self.goal_calls: list[tuple[str, int, int]] = []
        self.term_goal_calls: list[tuple[str, int, int]] = []

    def open_file(self, rel_path: str) -> None:
        _ = rel_path

    def get_diagnostics(self, rel_path: str):
        _ = rel_path
        self.diagnostics_called = True
        return []

    def get_file_content(self, rel_path: str) -> str:
        _ = rel_path
        return "theorem demo (n : Nat) : n = n := by\n  rfl\n"

    def _local_to_uri(self, rel_path: str) -> str:
        return f"file:///{rel_path}"

    def get_goal(self, rel_path: str, line: int, column: int):
        self.goal_calls.append((rel_path, line, column))
        return {"goals": ["⊢ fallback goal"]}

    def get_term_goal(self, rel_path: str, line: int, column: int):
        self.term_goal_calls.append((rel_path, line, column))
        return {"goal": "```lean\nNat\n```"}


def _make_ctx(client: FakeClient) -> types.SimpleNamespace:
    lifespan_context = types.SimpleNamespace(client=client)
    request_context = types.SimpleNamespace(lifespan_context=lifespan_context)
    return types.SimpleNamespace(request_context=request_context)


def test_interactive_goals_warms_diagnostics_before_rpc(
    monkeypatch,
) -> None:
    client = FakeClient()
    ctx = _make_ctx(client)
    rel_path = "InteractiveGoalSample.lean"

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _file: rel_path)

    def fake_rpc_call(*args, **kwargs):
        _ = args, kwargs
        assert client.diagnostics_called
        return {
            "goals": [
                {
                    "hyps": [{"names": ["n"], "type": {"text": "Nat"}}],
                    "type": {"text": "n = n"},
                }
            ]
        }

    monkeypatch.setattr(server, "_rpc_call_with_retry", fake_rpc_call)

    result = server.interactive_goals(
        ctx=ctx,
        file_path="/tmp/InteractiveGoalSample.lean",
        line=2,
        column=3,
    )

    assert result.rendered
    assert "⊢ n = n" in result.rendered_text


def test_interactive_goals_fallbacks_to_lsp_on_missing_rpc_method(
    monkeypatch,
) -> None:
    client = FakeClient()
    ctx = _make_ctx(client)
    rel_path = "InteractiveGoalSample.lean"

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _file: rel_path)

    def fake_rpc_call(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("RpcMethodNotFound")

    monkeypatch.setattr(server, "_rpc_call_with_retry", fake_rpc_call)

    result = server.interactive_goals(
        ctx=ctx,
        file_path="/tmp/InteractiveGoalSample.lean",
        line=2,
        column=3,
    )

    assert result.goals == []
    assert result.rendered == ["⊢ fallback goal"]
    assert result.rendered_text == "⊢ fallback goal"
    assert client.goal_calls == [(rel_path, 1, 2)]


def test_interactive_term_goal_fallbacks_to_lsp_on_missing_rpc_method(
    monkeypatch,
) -> None:
    client = FakeClient()
    ctx = _make_ctx(client)
    rel_path = "InteractiveGoalSample.lean"

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _file: rel_path)

    def fake_rpc_call(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("method not found")

    monkeypatch.setattr(server, "_rpc_call_with_retry", fake_rpc_call)

    result = server.interactive_term_goal(
        ctx=ctx,
        file_path="/tmp/InteractiveGoalSample.lean",
        line=2,
        column=3,
    )

    assert result.rendered == "Nat"
    assert result.goal == {"goal": "```lean\nNat\n```"}
    assert client.term_goal_calls == [(rel_path, 1, 2)]
