from __future__ import annotations

import types
from pathlib import Path

import pytest
from mcp.types import ResourceLink

from lean_lsp_mcp import server


def test_lean_file_resource_round_trip_with_encoded_path(tmp_path: Path) -> None:
    lean_file = tmp_path / "My Sample.lean"
    lean_file.write_text("def sample : Nat := 1\n", encoding="utf-8")

    uri = server._lean_file_resource_uri(lean_file)
    encoded_path = uri.removeprefix("lean://file/")

    content = server.lean_file_resource(encoded_path)
    assert content == "def sample : Nat := 1\n"


def test_lean_file_resource_requires_absolute_path() -> None:
    with pytest.raises(ValueError, match="absolute Lean file path"):
        server.lean_file_resource("relative/path.lean")


def test_declaration_file_includes_resource_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declaration_file = tmp_path / "NatDecl.lean"
    declaration_file.write_text("def NatDecl : Nat := 0\n", encoding="utf-8")

    class FakeClient:
        def open_file(self, rel_path: str) -> None:
            _ = rel_path

        def get_file_content(self, rel_path: str) -> str:
            _ = rel_path
            return "def foo : Nat := Nat.succ 0\n"

        def get_declarations(self, rel_path: str, line: int, column: int):
            _ = rel_path, line, column
            return [{"uri": "file:///dummy/NatDecl.lean"}]

        def _uri_to_abs(self, uri: str) -> str:
            _ = uri
            return str(declaration_file)

    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(
            lifespan_context=types.SimpleNamespace(client=FakeClient())
        )
    )

    monkeypatch.setattr(server, "setup_client_for_file", lambda _ctx, _file: "Foo.lean")
    monkeypatch.setattr(
        server,
        "find_start_position",
        lambda _content, _symbol: {"line": 0, "column": 0},
    )

    result = server.declaration_file(
        ctx=ctx,
        file_path=str(declaration_file),
        symbol="Nat",
    )

    assert isinstance(result, list)
    assert len(result) == 2
    link = result[1]
    assert isinstance(link, ResourceLink)
    assert str(link.uri) == server._lean_file_resource_uri(declaration_file)
