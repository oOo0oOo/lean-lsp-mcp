from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


@pytest.mark.asyncio
async def test_highlight_occurrences(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    target_file = test_project_path / "GoalSample.lean"

    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_highlight_occurrences",
            {
                "file_path": str(target_file),
                "line": 1,
                "column": 1,
                "query": "Nat",
                "text": "Nat Nat",
            },
        )
        structured = result.structuredContent
        assert structured is not None
        rendered = structured.get("rendered_text")
        assert isinstance(rendered, str)
        assert "Nat" in rendered
