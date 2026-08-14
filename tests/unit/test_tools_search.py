"""Unit tests for the symbol index enrichment behind lean_local_search."""

from pathlib import Path

import pytest

from lean_lsp_mcp.file_utils import AllowedPathRoot, LeanPathPolicy
from lean_lsp_mcp.tools import search as search_tool


SOURCE_MATCHES = [{"name": "Ns.thing_long", "kind": "theorem", "file": "A.lean"}]


def _policy(project_root: Path) -> LeanPathPolicy:
    return LeanPathPolicy(
        project_root=project_root,
        allowed_roots=(AllowedPathRoot(project_root, ""),),
    )


class _FakeClient:
    """Stands in for a running AsyncLeanLSPClient."""

    def __init__(self, symbols=None, error=None):
        self.symbols = symbols or []
        self.error = error
        self.calls = []

    async def workspace_symbol(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.symbols, True


async def test_source_results_are_returned_when_no_client_is_running(
    monkeypatch, tmp_path
):
    """No language server means no enrichment, and no `lake serve` boot either."""
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: None)

    result = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert result == SOURCE_MATCHES


async def test_index_matches_are_merged_when_a_client_is_running(monkeypatch, tmp_path):
    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    client = _FakeClient(
        symbols=[{"name": "Ns.thing", "location": {"uri": declaration.as_uri()}}]
    )
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    # The exact match exists only in the index, and still sorts first.
    assert [match["name"] for match in result] == ["Ns.thing", "Ns.thing_long"]

    query, kwargs = client.calls[0]
    assert query == "thing"
    assert kwargs["max_results"] == 10
    # Waiting for the index would stall a search that is meant to be fast.
    assert kwargs["wait_for_index"] == 0.0


@pytest.mark.parametrize(
    "error", [RuntimeError("index unavailable"), TimeoutError("workspace/symbol")]
)
async def test_a_failing_symbol_query_does_not_break_the_search(
    error, monkeypatch, tmp_path
):
    client = _FakeClient(error=error)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert result == SOURCE_MATCHES
