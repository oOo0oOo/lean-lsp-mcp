from __future__ import annotations

import types

import pytest

from lean_lsp_mcp import server


class _DummyWidgetClient:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.widget_payload: dict | None = None
        self.widgets_response: list | None = []
        self.interactive_response: list = []

    def open_file(self, rel_path: str) -> None:
        self.opened.append(rel_path)

    def get_widgets(self, rel_path: str, line: int, column: int):  # pragma: no cover
        return self.widgets_response

    def get_widget_source(self, rel_path: str, line: int, column: int, widget: dict):
        self.widget_payload = dict(widget)
        return {"sourcetext": "export default 1"}

    def get_interactive_diagnostics(
        self,
        rel_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        extract_widgets: bool = False,
    ):
        return self.interactive_response


def _make_ctx(client: _DummyWidgetClient) -> types.SimpleNamespace:
    lifespan = types.SimpleNamespace(client=client)
    request_context = types.SimpleNamespace(lifespan_context=lifespan)
    return types.SimpleNamespace(request_context=request_context)


def test_widget_source_accepts_string_javascript_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    result = server.widget_source(
        ctx=ctx,
        file_path="/tmp/Foo.lean",
        line=1,
        column=1,
        javascript_hash=" abc123 ",
    )

    assert result.sourcetext == "export default 1"
    assert client.widget_payload is not None
    assert client.widget_payload["javascriptHash"] == "abc123"


def test_widget_source_accepts_snake_case_hash_from_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    result = server.widget_source(
        ctx=ctx,
        file_path="/tmp/Foo.lean",
        line=1,
        column=1,
        widget={"id": "w", "javascript_hash": " local-hash "},
    )

    assert result.sourcetext == "export default 1"
    assert client.widget_payload is not None
    assert client.widget_payload["javascriptHash"] == "local-hash"


def test_widget_source_requires_non_empty_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    with pytest.raises(server.LeanToolError, match="non-empty javascriptHash"):
        server.widget_source(
            ctx=ctx,
            file_path="/tmp/Foo.lean",
            line=1,
            column=1,
            widget={"id": "w"},
        )


def test_widgets_filters_non_dict_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _DummyWidgetClient()
    client.widgets_response = [
        {"id": "w1", "javascriptHash": "h1", "props": {"x": 1}},
        "bad-payload",
    ]
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    result = server.widgets(
        ctx=ctx,
        file_path="/tmp/Foo.lean",
        line=1,
        column=1,
    )
    assert len(result.items) == 1
    assert result.items[0].id == "w1"


def test_interactive_diagnostics_requires_range_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    with pytest.raises(server.LeanToolError, match="both start_line and end_line"):
        server.interactive_diagnostics(
            ctx=ctx,
            file_path="/tmp/Foo.lean",
            start_line=3,
        )


def test_interactive_diagnostics_rejects_inverted_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    with pytest.raises(server.LeanToolError, match="start_line must be <="):
        server.interactive_diagnostics(
            ctx=ctx,
            file_path="/tmp/Foo.lean",
            start_line=5,
            end_line=4,
        )


def test_interactive_diagnostics_filters_non_dict_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _DummyWidgetClient()
    client.interactive_response = [{"severity": 1, "message": "ok"}, "junk"]
    ctx = _make_ctx(client)
    monkeypatch.setattr(server, "setup_client_for_file", lambda *_: "Foo.lean")

    result = server.interactive_diagnostics(
        ctx=ctx,
        file_path="/tmp/Foo.lean",
        extract_widgets=False,
    )

    assert len(result.diagnostics) == 1
    assert result.widgets == []
