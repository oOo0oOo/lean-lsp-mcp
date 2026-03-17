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
            },
        )
        refs_text = result_text(refs)

        # Should find the definition + at least 2 usages
        # At least 3 references: def, usesHelper, helperIsFortyTwo
        assert refs_text.count("line") >= 3


