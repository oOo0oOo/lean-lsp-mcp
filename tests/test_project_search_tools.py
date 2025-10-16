from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


def _first_json_block(result) -> dict[str, str]:
    for block in result.content:
        text = getattr(block, "text", "").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    pytest.skip("Tool did not return JSON content")


@pytest.fixture()
def goal_file(test_project_path: Path) -> Path:
    goal_path = test_project_path / "GoalSample.lean"
    goal_path.write_text(
        """import Mathlib

theorem sample_goal : True := by
  trivial
""",
        encoding="utf-8",
    )
    return goal_path


@pytest.mark.asyncio
async def test_leansearch_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_leansearch",
            {"query": "Nat.succ"},
        )
        entry = _first_json_block(result)
        assert {"module_name", "name", "type"} <= set(entry.keys())


@pytest.mark.asyncio
async def test_loogle_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_loogle",
            {"query": "Nat"},
        )
        entry = _first_json_block(result)
        assert {"module", "name", "type"} <= set(entry.keys())


@pytest.mark.asyncio
async def test_state_search_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    goal_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        goal_result = await client.call_tool(
            "lean_goal",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        assert "⊢ True" in result_text(goal_result)

        state_search = await client.call_tool(
            "lean_state_search",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        text = result_text(state_search)
        assert "Results for line" in text


@pytest.mark.asyncio
async def test_hammer_premise_tool(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    goal_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_goal",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        hammer = await client.call_tool(
            "lean_hammer_premise",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        assert "Results for line" in result_text(hammer)
