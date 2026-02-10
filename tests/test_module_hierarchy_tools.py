from __future__ import annotations

from pathlib import Path
from typing import AsyncContextManager, Callable

import pytest

from tests.helpers.mcp_client import MCPClient


def _mathlib_file(test_project_path: Path) -> Path:
    candidate = (
        test_project_path
        / ".lake"
        / "packages"
        / "mathlib"
        / "Mathlib"
        / "Data"
        / "List"
        / "Basic.lean"
    )
    if not candidate.exists():
        pytest.skip(
            "mathlib sources were not downloaded; run `lake update` inside tests/test_project."
        )
    return candidate


@pytest.mark.asyncio
async def test_module_hierarchy_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    target_file = _mathlib_file(test_project_path)

    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_imports",
            {
                "file_path": str(target_file),
                "lean_project_path": str(test_project_path),
                "include_imports": True,
                "include_imported_by": True,
                "view": "graph",
                "direction": "both",
                "depth": 1,
                "max_nodes": 64,
            },
        )

        structured = result.structuredContent
        assert structured is not None, "Expected structured content from tool"

        module = structured.get("module")
        assert isinstance(module, dict), "Expected module info in response"
        assert module.get("name", "").startswith("Mathlib."), (
            f"Unexpected module name: {module}"
        )

        imports = structured.get("imports")
        assert isinstance(imports, list), "Expected imports list"
        assert imports, "Expected at least one import"
        first_import = imports[0]
        assert "module" in first_import and "kind" in first_import
        assert "metaKind" in first_import["kind"]

        imported_by = structured.get("imported_by")
        assert isinstance(imported_by, list), "Expected imported_by list"

        graph = structured.get("graph")
        assert isinstance(graph, dict), "Expected graph view payload"
        assert isinstance(graph.get("nodes"), list), "Expected graph nodes list"
        assert isinstance(graph.get("edges"), list), "Expected graph edges list"
        assert graph.get("root") == module.get("name"), "Expected root module in graph"

        tree_result = await client.call_tool(
            "lean_imports",
            {
                "file_path": str(target_file),
                "lean_project_path": str(test_project_path),
                "view": "tree",
                "direction": "imports",
                "depth": 1,
                "max_nodes": 32,
            },
        )
        tree_structured = tree_result.structuredContent
        assert tree_structured is not None, "Expected tree structured content"
        tree = tree_structured.get("tree")
        assert isinstance(tree, dict), "Expected tree view payload"
        assert tree.get("name", "").startswith("Mathlib."), (
            f"Unexpected tree root: {tree}"
        )
        assert isinstance(tree.get("children"), list), "Expected tree children list"

        # Deprecated alias should still work for compatibility.
        alias_result = await client.call_tool(
            "lean_module_hierarchy",
            {
                "file_path": str(target_file),
                "lean_project_path": str(test_project_path),
                "include_imports": True,
                "include_imported_by": False,
            },
        )
        alias_structured = alias_result.structuredContent
        assert alias_structured is not None
        assert isinstance(alias_structured.get("imports"), list)
