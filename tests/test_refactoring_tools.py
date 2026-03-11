from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture()
def refactor_file(test_project_path: Path) -> Path:
    path = test_project_path / "RefactorTools.lean"
    path.write_text(
        textwrap.dedent(
            """
            import Mathlib

            def myHelper : Nat := 42

            def usesHelper : Nat := myHelper + 1

            theorem helperIsFortyTwo : myHelper = 42 := rfl
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_find_references(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    refactor_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        # Find references to myHelper (defined on line 3, column 5)
        refs = await client.call_tool(
            "lean_references",
            {
                "file_path": str(refactor_file),
                "line": 3,
                "column": 5,
                "include_declaration": True,
            },
        )
        refs_text = result_text(refs)

        # Should find the definition + at least 2 usages
        assert "myHelper" in refs_text
        # At least 3 references: def, usesHelper, helperIsFortyTwo
        assert refs_text.count("line") >= 3


@pytest.mark.asyncio
async def test_find_references_no_declaration(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    refactor_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        refs = await client.call_tool(
            "lean_references",
            {
                "file_path": str(refactor_file),
                "line": 3,
                "column": 5,
                "include_declaration": False,
            },
        )
        refs_text = result_text(refs)
        # With include_declaration=False, should have fewer results
        assert "myHelper" in refs_text


@pytest.mark.asyncio
async def test_rename_symbol(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    refactor_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_rename",
            {
                "file_path": str(refactor_file),
                "line": 3,
                "column": 5,
                "new_name": "myRenamedHelper",
            },
        )
        result_str = result_text(result)
        assert "myHelper" in result_str
        assert "myRenamedHelper" in result_str

        # Verify file was updated on disk
        updated_content = refactor_file.read_text(encoding="utf-8")
        assert "myRenamedHelper" in updated_content
        assert "def myHelper" not in updated_content
        # All references should be renamed
        assert "myHelper" not in updated_content
