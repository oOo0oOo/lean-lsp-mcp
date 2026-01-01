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
            "lean_module_hierarchy",
            {
                "file_path": str(target_file),
                "lean_project_path": str(test_project_path),
                "include_imports": True,
                "include_imported_by": True,
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
