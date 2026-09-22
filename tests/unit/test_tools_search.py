"""Unit tests for the symbol index enrichment behind lean_local_search."""

from pathlib import Path

import pytest

from lean_lsp_mcp.file_utils import AllowedPathRoot, LeanPathPolicy
from lean_lsp_mcp.models import IndexStatus
from lean_lsp_mcp.tools import search as search_tool


SOURCE_MATCHES = [{"name": "Ns.thing_long", "kind": "theorem", "file": "A.lean"}]


def _policy(project_root: Path) -> LeanPathPolicy:
    return LeanPathPolicy(
        project_root=project_root,
        allowed_roots=(AllowedPathRoot(project_root, ""),),
    )


class _FakeClient:
    """Stands in for a running AsyncLeanLSPClient."""

    def __init__(self, symbols=None, error=None, index_ready=True):
        self.symbols = symbols or []
        self.error = error
        self.index_ready = index_ready
        self.calls = []

    async def workspace_symbol(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.symbols, self.index_ready


async def test_source_results_are_returned_when_no_client_is_running(
    monkeypatch, tmp_path
):
    """No language server means no enrichment, and no `lake serve` boot either."""
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: None)

    result, index = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert result == SOURCE_MATCHES
    assert index is IndexStatus.unavailable


async def test_index_matches_are_merged_when_a_client_is_running(monkeypatch, tmp_path):
    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    client = _FakeClient(
        symbols=[{"name": "Ns.thing", "location": {"path": str(declaration)}}]
    )
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result, index = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    # The exact match exists only in the index, and still sorts first.
    assert [match["name"] for match in result] == ["Ns.thing", "Ns.thing_long"]
    assert index is IndexStatus.consulted

    query, kwargs = client.calls[0]
    assert query == "thing"
    assert kwargs["max_results"] == 10
    # Waiting for the index would stall a search that is meant to be fast.
    assert kwargs["wait_for_index"] == 0.0
    assert kwargs["timeout"] == 2.0


@pytest.mark.parametrize(
    "error", [RuntimeError("index unavailable"), TimeoutError("workspace/symbol")]
)
async def test_a_failing_symbol_query_does_not_break_the_search(
    error, monkeypatch, tmp_path
):
    client = _FakeClient(error=error)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result, index = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert result == SOURCE_MATCHES
    assert index is IndexStatus.error


async def test_a_partial_index_is_reported_as_warming_not_consulted(
    monkeypatch, tmp_path
):
    """The server says whether the index finished loading; discarding that
    would let the tool claim completeness for an index still being built.

    `warming` still carries whatever the index had, so the results are useful
    -- they are just not proof that anything is absent.
    """
    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    client = _FakeClient(
        symbols=[{"name": "Ns.thing", "location": {"path": str(declaration)}}],
        index_ready=False,
    )
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result, index = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert index is IndexStatus.warming
    assert [match["name"] for match in result] == ["Ns.thing", "Ns.thing_long"]


def test_compiler_helpers_are_kept_out_of_index_results(tmp_path) -> None:
    """`workspace/symbol` answers from the environment, which holds names the
    elaborator invented for itself.

    A macro-expansion auxiliary or a hygiene-renamed local means nothing to the
    caller and crowds real declarations out of a limited result window.
    """
    from lean_lsp_mcp.search_utils import workspace_symbol_matches

    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    symbols = [
        {"name": name, "location": {"path": str(declaration)}}
        for name in (
            "_private.Demo.BumpFunction.0._aux_Demo_macroRules_term_1",
            "_private.Demo.BumpFunction.0.«_aux_Demo_macroRules_term#_1»",
            "Demo.definition._@.Demo.Facts.123._hygCtx._hyg.2",
            "Demo.mk",
            "Demo._hyg",
            "_private.Demo.0.BumpFunction_theorem",
        )
    ]

    names = [
        match["name"] for match in workspace_symbol_matches(symbols, _policy(tmp_path))
    ]

    # Ordinary private declarations and names that merely look hygienic stay.
    assert names == ["Demo.mk", "Demo._hyg", "_private.Demo.0.BumpFunction_theorem"]


def test_a_helper_named_exactly_is_still_returned(tmp_path) -> None:
    """A search must never hide what it was asked for."""
    from lean_lsp_mcp.search_utils import workspace_symbol_matches

    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    helper = "Demo.definition._@.Demo.Facts.123._hygCtx._hyg.2"
    symbols = [{"name": helper, "location": {"path": str(declaration)}}]

    assert workspace_symbol_matches(symbols, _policy(tmp_path)) == []
    assert [
        match["name"]
        for match in workspace_symbol_matches(symbols, _policy(tmp_path), helper)
    ] == [helper]
    assert [
        match["name"]
        for match in workspace_symbol_matches(
            symbols, _policy(tmp_path), f"_root_.{helper}"
        )
    ] == [helper]
