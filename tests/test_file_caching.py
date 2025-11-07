"""Test file caching optimization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture()
def cache_test_file(test_project_path: Path) -> Path:
    path = test_project_path / "CacheTest.lean"
    content = """import Mathlib

def cachedValue : Nat := 42

theorem cachedTheorem : cachedValue = 42 := by rfl
"""
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_disk_changes_detected(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    cache_test_file: Path,
) -> None:
    """Disk changes must be detected and reprocessed correctly."""

    async with mcp_client_factory() as client:
        goal1 = await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        result1 = result_text(goal1)
        assert "no goals" in result1.lower()

        cache_test_file.write_text(
            """import Mathlib

def cachedValue : Nat := 42

theorem cachedTheorem : cachedValue = 42 := by sorry
""",
            encoding="utf-8",
        )

        goal2 = await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        result2 = result_text(goal2)

        assert "cachedValue = 42" in result2, (
            f"Should show goal at sorry, got: {result2}"
        )


@pytest.mark.asyncio
async def test_multiple_tools_share_file(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    cache_test_file: Path,
) -> None:
    """Different tools must reuse cached file state correctly."""

    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages", {"file_path": str(cache_test_file)}
        )
        await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        hover = await client.call_tool(
            "lean_hover_info",
            {"file_path": str(cache_test_file), "line": 3, "column": 5},
        )

        assert "cachedValue" in result_text(hover)
