from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


@pytest.fixture()
def interactive_goal_file(test_project_path: Path) -> Path:
    path = test_project_path / "InteractiveGoalSample.lean"
    if not path.exists():
        pytest.skip("Interactive goal sample file missing")
    return path


@pytest.mark.asyncio
async def test_interactive_goals(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    interactive_goal_file: Path,
) -> None:
    content = interactive_goal_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    goal_line_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("rfl")
    )
    column = lines[goal_line_index].index("rfl") + 1

    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_interactive_goals",
            {
                "file_path": str(interactive_goal_file),
                "line": goal_line_index + 1,
                "column": column,
            },
        )
        structured = result.structuredContent
        assert structured is not None, "Expected structured content"
        goals = structured.get("goals")
        rendered = structured.get("rendered")
        rendered_text = structured.get("rendered_text")
        assert isinstance(goals, list), "Expected goals list"
        assert isinstance(rendered, list), "Expected rendered list"
        assert isinstance(rendered_text, str), "Expected rendered_text"
        assert "⊢" in rendered_text
        assert "n" in rendered_text

        term_result = await client.call_tool(
            "lean_interactive_term_goal",
            {
                "file_path": str(interactive_goal_file),
                "line": goal_line_index + 1,
                "column": column,
            },
        )
        term_structured = term_result.structuredContent
        assert term_structured is not None, "Expected structured content"
        assert "rendered" in term_structured
