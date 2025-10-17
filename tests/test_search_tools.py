from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


def _first_json_block(result) -> dict[str, str] | None:
    for block in result.content:
        text = getattr(block, "text", "").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


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
async def test_search_tools(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    goal_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        loogle = await client.call_tool(
            "lean_loogle",
            {"query": "Nat"},
        )
        loogle_entry = _first_json_block(loogle)
        if loogle_entry is None:
            pytest.skip("lean_loogle did not return JSON content")
        assert {"module", "name", "type"} <= set(loogle_entry.keys())

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
        assert "Results for line" in text or "lean state search error" in text

        hammer = await client.call_tool(
            "lean_hammer_premise",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        assert "Results for line" in result_text(hammer)

        local_search = await client.call_tool(
            "lean_local_search",
            {
                "query": "sampleTheorem",
                "project_root": str(goal_file.parent),
            },
        )
        local_entry = _first_json_block(local_search)
        if local_entry is None:
            message = result_text(local_search).strip()
            if "ripgrep" in message.lower():
                pytest.skip(message)
            pytest.fail(f"lean_local_search returned unexpected content: {message}")
        assert local_entry == {
            "name": "sampleTheorem",
            "kind": "theorem",
            "file": "EditorTools.lean",
        }

        leansearch = await client.call_tool(
            "lean_leansearch",
            {"query": "Nat.succ"},
        )
        entry = _first_json_block(leansearch)
        if entry is None:
            pytest.skip("lean_leansearch did not return JSON content")
        assert {"module_name", "name", "type"} <= set(entry.keys())
