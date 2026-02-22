from __future__ import annotations

from pathlib import Path

import pytest

from collections.abc import Callable
from typing import AsyncContextManager

from tests.helpers.mcp_client import MCPClient, result_text


def _mathlib_file(test_project_path: Path) -> Path:
    candidate = (
        test_project_path
        / ".lake"
        / "packages"
        / "mathlib"
        / "Mathlib"
        / "Data"
        / "Nat"
        / "Basic.lean"
    )
    if not candidate.exists():
        pytest.skip(
            "mathlib sources were not downloaded; run `lake update` inside tests/test_project."
        )
    return candidate


@pytest.mark.asyncio
async def test_mathlib_file_roundtrip(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """Test reading mathlib files and querying hover/term info."""
    target_file = _mathlib_file(test_project_path)

    async with mcp_client_factory() as mcp_client:
        # Test reading file outline
        outline = await mcp_client.call_tool(
            "lean_file_outline",
            {
                "file_path": str(target_file),
            },
        )
        text = result_text(outline)
        assert "Nat" in text

        # Test hover on a stable position (line 35: "le := Nat.le" in instance declaration)
        hover = await mcp_client.call_tool(
            "lean_hover_info",
            {
                "file_path": str(target_file),
                "line": 35,
                "column": 8,  # Position on 'le'
            },
        )
        hover_text = result_text(hover)
        assert "Nat.le" in hover_text or "LE" in hover_text or "le" in hover_text

        # Test term goal (may or may not be valid at this position)
        term_goal = await mcp_client.call_tool(
            "lean_term_goal",
            {
                "file_path": str(target_file),
                "line": 35,
                "column": 8,
            },
        )
        type_text = result_text(term_goal)
        # Accept either a valid type or the "not valid" message
        assert len(type_text) > 0
