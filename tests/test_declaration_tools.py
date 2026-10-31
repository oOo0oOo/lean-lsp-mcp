from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import orjson
import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture
def additive_reference_file(test_project_path: Path) -> Path:
    path = test_project_path / "DeclarationSourceTest.lean"
    path.write_text(
        "import Mathlib\n\n#check Finset.sum_range_succ\n", encoding="utf-8"
    )
    yield path
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_declaration_file_reports_additive_source_and_location(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    additive_reference_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_declaration_file",
            {
                "file_path": str(additive_reference_file),
                "symbol": "Finset.sum_range_succ",
                "context_lines": 0,
            },
        )

    declaration = orjson.loads(result_text(result))
    assert declaration["declaration_start_line"] is not None
    assert (
        declaration["declaration_end_line"] is None
        or declaration["declaration_end_line"] >= declaration["declaration_start_line"]
    )
    assert declaration["start_line"] == declaration["declaration_start_line"]
    assert "prod_range_succ" in declaration["content"]
    assert declaration["source_declaration"] == "Finset.prod_range_succ"
