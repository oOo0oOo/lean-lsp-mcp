from __future__ import annotations

import types

import pytest

from lean_lsp_mcp import server


class FakeClient:
    def open_file(self, rel_path: str) -> None:
        _ = rel_path

    def get_diagnostics(self, rel_path: str):
        _ = rel_path
        return []

    def get_file_content(self, rel_path: str) -> str:
        _ = rel_path
        return "theorem demo : True := by\n  trivial\n"

    def _local_to_uri(self, rel_path: str) -> str:
        return f"file:///{rel_path}"


def _make_ctx() -> types.SimpleNamespace:
    lifespan = types.SimpleNamespace(client=FakeClient())
    request_context = types.SimpleNamespace(lifespan_context=lifespan)
    return types.SimpleNamespace(request_context=request_context)


def test_highlight_occurrences_falls_back_when_widget_rpc_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _file: "Demo.lean"
    )

    def fake_rpc_call(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("method not found")

    monkeypatch.setattr(server, "_rpc_call_with_retry", fake_rpc_call)

    result = server.highlight_occurrences(
        ctx=_make_ctx(),
        file_path="/tmp/Demo.lean",
        line=2,
        column=3,
        query="nat",
        case_sensitive=False,
        text="Nat nat NAt",
    )

    assert result.rendered_text == "Nat nat NAt"
    assert result.highlighted_text == "❰Nat❱ ❰nat❱ ❰NAt❱"
    assert result.message == {"text": "Nat nat NAt"}


def test_highlight_occurrences_rejects_empty_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _file: "Demo.lean"
    )

    with pytest.raises(server.LeanToolError, match="must not be empty"):
        server.highlight_occurrences(
            ctx=_make_ctx(),
            file_path="/tmp/Demo.lean",
            line=2,
            column=3,
            query="   ",
            text="Nat nat",
        )


def test_highlight_occurrences_falls_back_when_rpc_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "setup_client_for_file", lambda _ctx, _file: "Demo.lean"
    )
    monkeypatch.setattr(server, "_rpc_call_with_retry", lambda *args, **kwargs: None)

    result = server.highlight_occurrences(
        ctx=_make_ctx(),
        file_path="/tmp/Demo.lean",
        line=2,
        column=3,
        query="Nat",
        text="Nat Nat",
    )

    assert result.rendered_text == "Nat Nat"
    assert result.highlighted_text == "❰Nat❱ ❰Nat❱"
