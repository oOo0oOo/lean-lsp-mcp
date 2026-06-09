"""Unit tests for the resolve_name helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lean_lsp_mcp.resolve import _match_name, resolve_name
from lean_lsp_mcp.utils import LeanToolError


# ---------------------------------------------------------------------------
# _match_name pure logic
# ---------------------------------------------------------------------------


class TestMatchName:
    def test_exact_full_name(self):
        assert _match_name("NFGame.IsNash", "NFGame.IsNash") is True

    def test_exact_short_name(self):
        assert _match_name("NFGame.IsNash", "IsNash") is True

    def test_suffix_match(self):
        assert _match_name("A.B.IsNash", "B.IsNash") is True

    def test_no_match(self):
        assert _match_name("NFGame.IsNash", "NotNash") is False

    def test_partial_short_no_match(self):
        assert _match_name("NFGame.IsNash", "Nash") is False

    def test_case_sensitive(self):
        assert _match_name("NFGame.IsNash", "isnash") is False

    def test_single_component_name(self):
        assert _match_name("sampleValue", "sampleValue") is True

    def test_short_matches_single_component(self):
        # "sampleValue" has no dot, so short = "sampleValue"
        assert _match_name("sampleValue", "sampleValue") is True


# ---------------------------------------------------------------------------
# resolve_name with mocked dependencies
# ---------------------------------------------------------------------------


@dataclass
class FakeLifespan:
    lean_project_path: Path | None = None
    client: Any = None


def _make_ctx(project_path: Path | None = None, client: Any = None) -> MagicMock:
    """Create a minimal mock Context with lifespan."""
    ctx = MagicMock()
    lifespan = FakeLifespan(lean_project_path=project_path, client=client)
    ctx.request_context.lifespan_context = lifespan
    return ctx


@pytest.mark.asyncio
async def test_resolve_empty_name():
    ctx = _make_ctx(project_path=Path("/proj"))
    with pytest.raises(LeanToolError, match="Name must not be empty"):
        await resolve_name(ctx, "")


@pytest.mark.asyncio
async def test_resolve_no_project_path():
    ctx = _make_ctx(project_path=None)
    with pytest.raises(LeanToolError, match="project path not set"):
        await resolve_name(ctx, "foo")


@pytest.mark.asyncio
async def test_resolve_not_found(monkeypatch):
    ctx = _make_ctx(project_path=Path("/proj"))

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [],
    )

    with pytest.raises(LeanToolError, match="not found"):
        await resolve_name(ctx, "NonexistentDecl")


@pytest.mark.asyncio
async def test_resolve_no_exact_match(monkeypatch):
    ctx = _make_ctx(project_path=Path("/proj"))

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "fooBar", "kind": "def", "file": "A.lean"},
            {"name": "fooBaz", "kind": "def", "file": "B.lean"},
        ],
    )

    with pytest.raises(LeanToolError, match="No exact match"):
        await resolve_name(ctx, "foo")


@pytest.mark.asyncio
async def test_resolve_exact_fqn(monkeypatch):
    mock_client = MagicMock()
    ctx = _make_ctx(project_path=Path("/proj"), client=mock_client)

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "Ns.myDef", "kind": "def", "file": "Ns.lean"},
        ],
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.setup_client_for_file",
        lambda c, path: "Ns.lean",
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.get_declaration_range",
        lambda client, rel, name: (5, 10),
    )

    result = await resolve_name(ctx, "Ns.myDef")

    assert result.full_name == "Ns.myDef"
    assert result.kind == "def"
    assert result.rel_path == "Ns.lean"
    assert result.start_line == 5
    assert result.end_line == 10


@pytest.mark.asyncio
async def test_resolve_short_name_unique(monkeypatch):
    mock_client = MagicMock()
    ctx = _make_ctx(project_path=Path("/proj"), client=mock_client)

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "Ns.myDef", "kind": "def", "file": "Ns.lean"},
        ],
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.setup_client_for_file",
        lambda c, path: "Ns.lean",
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.get_declaration_range",
        lambda client, rel, name: (5, 10),
    )

    result = await resolve_name(ctx, "myDef")

    assert result.full_name == "Ns.myDef"
    assert result.start_line == 5


@pytest.mark.asyncio
async def test_resolve_ambiguous_short_name(monkeypatch):
    ctx = _make_ctx(project_path=Path("/proj"))

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "A.myDef", "kind": "def", "file": "A.lean"},
            {"name": "B.myDef", "kind": "def", "file": "B.lean"},
        ],
    )

    with pytest.raises(LeanToolError, match="Ambiguous"):
        await resolve_name(ctx, "myDef")


@pytest.mark.asyncio
async def test_resolve_prefers_project_over_lake(monkeypatch):
    mock_client = MagicMock()
    ctx = _make_ctx(project_path=Path("/proj"), client=mock_client)

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "myDef", "kind": "def", "file": ".lake/packages/lib/Lib.lean"},
            {"name": "myDef", "kind": "def", "file": "MyModule.lean"},
        ],
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.setup_client_for_file",
        lambda c, path: "MyModule.lean",
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.get_declaration_range",
        lambda client, rel, name: (1, 3),
    )

    result = await resolve_name(ctx, "myDef")

    assert result.rel_path == "MyModule.lean"


@pytest.mark.asyncio
async def test_resolve_fqn_disambiguates(monkeypatch):
    mock_client = MagicMock()
    ctx = _make_ctx(project_path=Path("/proj"), client=mock_client)

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "A.myDef", "kind": "def", "file": "A.lean"},
            {"name": "B.myDef", "kind": "def", "file": "B.lean"},
        ],
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.setup_client_for_file",
        lambda c, path: "A.lean",
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.get_declaration_range",
        lambda client, rel, name: (1, 5),
    )

    result = await resolve_name(ctx, "A.myDef")

    assert result.full_name == "A.myDef"
    assert result.rel_path == "A.lean"


@pytest.mark.asyncio
async def test_resolve_lsp_range_not_found(monkeypatch):
    mock_client = MagicMock()
    ctx = _make_ctx(project_path=Path("/proj"), client=mock_client)

    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.lean_local_search",
        lambda query, limit, project_root: [
            {"name": "myDef", "kind": "def", "file": "A.lean"},
        ],
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.setup_client_for_file",
        lambda c, path: "A.lean",
    )
    monkeypatch.setattr(
        "lean_lsp_mcp.resolve.get_declaration_range",
        lambda client, rel, name: None,
    )

    with pytest.raises(LeanToolError, match="could not locate"):
        await resolve_name(ctx, "myDef")
