"""Unit tests for the lean_references tool."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean_lsp_mcp import server
from lean_lsp_mcp.models import ReferenceLocation, ReferencesResult


def _make_ctx(client: MagicMock) -> types.SimpleNamespace:
    context = server.AppContext(
        lean_project_path=Path("/fake/project"),
        client=client,
        rate_limit={"test": []},
        lean_search_available=True,
    )
    request_context = types.SimpleNamespace(lifespan_context=context)
    return types.SimpleNamespace(request_context=request_context)


def _make_raw_ref(uri: str, start_line: int, start_char: int, end_line: int, end_char: int) -> dict:
    return {
        "uri": uri,
        "range": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
    }


@patch("lean_lsp_mcp.server.setup_client_for_file", return_value="Test.lean")
def test_references_parses_lsp_response(mock_setup: MagicMock) -> None:
    client = MagicMock()
    client._uri_to_abs.return_value = Path("/fake/project/Test.lean")
    client.get_references.return_value = [
        _make_raw_ref("file:///fake/project/Test.lean", 2, 4, 2, 12),
        _make_raw_ref("file:///fake/project/Test.lean", 4, 24, 4, 32),
        _make_raw_ref("file:///fake/project/Test.lean", 6, 30, 6, 38),
    ]

    ctx = _make_ctx(client)
    result = server.references(ctx, "/fake/project/Test.lean", 3, 5)

    assert isinstance(result, ReferencesResult)
    assert len(result.items) == 3
    # LSP 0-indexed -> 1-indexed
    assert result.items[0].line == 3
    assert result.items[0].column == 5
    assert result.items[0].end_line == 3
    assert result.items[0].end_column == 13
    client.get_references.assert_called_once_with("Test.lean", 2, 4, include_declaration=True)


@patch("lean_lsp_mcp.server.setup_client_for_file", return_value="Test.lean")
def test_references_empty_result(mock_setup: MagicMock) -> None:
    client = MagicMock()
    client.get_references.return_value = None

    ctx = _make_ctx(client)
    result = server.references(ctx, "/fake/project/Test.lean", 1, 1)

    assert isinstance(result, ReferencesResult)
    assert len(result.items) == 0


@patch("lean_lsp_mcp.server.setup_client_for_file", return_value=None)
def test_references_invalid_path(mock_setup: MagicMock) -> None:
    client = MagicMock()
    ctx = _make_ctx(client)

    with pytest.raises(server.LeanToolError, match="Invalid Lean file path"):
        server.references(ctx, "/nonexistent/Test.lean", 1, 1)


@patch("lean_lsp_mcp.server.setup_client_for_file", return_value="Test.lean")
def test_references_lsp_exception(mock_setup: MagicMock) -> None:
    client = MagicMock()
    client.get_references.side_effect = RuntimeError("LSP timeout")

    ctx = _make_ctx(client)

    with pytest.raises(server.LeanToolError, match="Failed to get references"):
        server.references(ctx, "/fake/project/Test.lean", 1, 1)
