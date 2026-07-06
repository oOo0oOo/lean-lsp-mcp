"""Unit tests for the lean_references tool (async client API)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from lean_lsp_mcp import server
from lean_lsp_mcp.models import ReferencesResult


def _make_project(root: Path) -> Path:
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (root / "lakefile.toml").write_text('name = "test"\n')
    return root


class _FakeClient:
    """Minimal async-client stand-in for the references tool."""

    def __init__(self, project_path: Path, refs=None, exc: Exception | None = None):
        self.project_path = str(project_path)
        self._refs = refs or []
        self._exc = exc
        self.calls: list[tuple] = []

    async def reload_from_disk(self, path: str, wait: bool = False):
        return types.SimpleNamespace(text="")

    async def references(
        self,
        path: str,
        line: int,
        col: int,
        include_declaration: bool = True,
        max_results=None,
        fresh: bool = True,
    ):
        self.calls.append((path, line, col, include_declaration))
        if self._exc is not None:
            raise self._exc
        return list(self._refs)


def _make_ctx(client, project_root: Path) -> types.SimpleNamespace:
    context = server.AppContext(
        lean_project_path=project_root,
        client=client,
        rate_limit={"test": []},
        lean_search_available=True,
    )
    request_context = types.SimpleNamespace(lifespan_context=context)
    return types.SimpleNamespace(request_context=request_context)


def _patch_setup(monkeypatch: pytest.MonkeyPatch, rel_path: str | None) -> None:
    async def fake_setup(_ctx, _path):
        return rel_path

    monkeypatch.setattr(server, "setup_client_for_file", fake_setup)


def _ref(path: str, start_line: int, start_char: int, end_line: int, end_char: int):
    # Shape produced by AsyncLeanLSPClient.references(): codepoint range +
    # project-relative "path".
    return {
        "path": path,
        "range": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
    }


@pytest.mark.asyncio
async def test_references_parses_lsp_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "Test.lean"
    lean_file.write_text("def x := 0\n")
    client = _FakeClient(
        project,
        refs=[
            _ref("Test.lean", 2, 4, 2, 12),
            _ref("Test.lean", 4, 24, 4, 32),
            _ref("Test.lean", 6, 30, 6, 38),
        ],
    )
    _patch_setup(monkeypatch, "Test.lean")

    ctx = _make_ctx(client, project)
    result = await server.references(ctx, str(lean_file), 3, 5)

    assert isinstance(result, ReferencesResult)
    assert len(result.items) == 3
    assert result.total == 3
    # 0-indexed -> 1-indexed
    assert result.items[0].line == 3
    assert result.items[0].column == 5
    assert result.items[0].end_line == 3
    assert result.items[0].end_column == 13
    assert client.calls == [("Test.lean", 2, 4, True)]


@pytest.mark.asyncio
async def test_references_truncates_to_max_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "Test.lean"
    lean_file.write_text("def x := 0\n")
    client = _FakeClient(
        project, refs=[_ref("Test.lean", i, 0, i, 4) for i in range(10)]
    )
    _patch_setup(monkeypatch, "Test.lean")

    ctx = _make_ctx(client, project)
    result = await server.references(ctx, str(lean_file), 1, 1, max_results=3)

    assert len(result.items) == 3
    assert result.total == 10


@pytest.mark.asyncio
async def test_references_empty_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "Test.lean"
    lean_file.write_text("def x := 0\n")
    client = _FakeClient(project, refs=[])
    _patch_setup(monkeypatch, "Test.lean")

    ctx = _make_ctx(client, project)
    result = await server.references(ctx, str(lean_file), 1, 1)

    assert isinstance(result, ReferencesResult)
    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_references_invalid_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    client = _FakeClient(project)
    _patch_setup(monkeypatch, None)

    ctx = _make_ctx(client, project)
    with pytest.raises(server.LeanToolError, match="Invalid Lean file path"):
        await server.references(ctx, "/nonexistent/Test.lean", 1, 1)


@pytest.mark.asyncio
async def test_references_lsp_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _make_project(tmp_path / "proj")
    lean_file = project / "Test.lean"
    lean_file.write_text("def x := 0\n")
    client = _FakeClient(project, exc=RuntimeError("LSP timeout"))
    _patch_setup(monkeypatch, "Test.lean")

    ctx = _make_ctx(client, project)
    with pytest.raises(server.LeanToolError, match="Failed to get references"):
        await server.references(ctx, str(lean_file), 1, 1)
