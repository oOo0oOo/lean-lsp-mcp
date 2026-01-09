from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import orjson
import pytest

from tests.helpers.mcp_client import MCPClient, result_text


def _first_result_item(result) -> dict[str, str] | None:
    """Extract the first item from a result that returns a list wrapped in {"items": [...]}.

    Handles both structured content and text content formats.
    """
    # Try structured content first (new format)
    if result.structuredContent is not None:
        items = result.structuredContent.get("items", [])
        return items[0] if items else None

    # Fall back to parsing text content
    for block in result.content:
        text = getattr(block, "text", "").strip()
        if not text:
            continue
        try:
            parsed = orjson.loads(text)
            # Handle {"items": [...]} wrapper format
            if isinstance(parsed, dict) and "items" in parsed:
                items = parsed["items"]
                return items[0] if items else None
            # Handle bare list format (legacy)
            if isinstance(parsed, list):
                return parsed[0] if parsed else None
            return parsed
        except orjson.JSONDecodeError:
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
        loogle_entry = _first_result_item(loogle)
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
            expect_error=True,
        )
        # Now returns JSON array of StateSearchResult models
        state_entry = _first_result_item(state_search)
        if state_entry is not None:
            assert "name" in state_entry

        hammer = await client.call_tool(
            "lean_hammer_premise",
            {
                "file_path": str(goal_file),
                "line": 4,
                "column": 3,
            },
        )
        # Now returns JSON array of PremiseResult models
        hammer_entry = _first_result_item(hammer)
        if hammer_entry is not None:
            assert "name" in hammer_entry

        local_search = await client.call_tool(
            "lean_local_search",
            {
                "query": "sampleTheorem",
                "project_root": str(goal_file.parent),
            },
        )
        local_entry = _first_result_item(local_search)
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
        entry = _first_result_item(leansearch)
        if entry is None:
            pytest.skip("lean_leansearch did not return JSON content")
        assert {"module_name", "name", "type"} <= set(entry.keys())

        # Test lean_finder with different query types
        finder_informal = await client.call_tool(
            "lean_leanfinder",
            {
                "query": "If two algebraic elements have the same minimal polynomial, are they related by a field isomorphism?",
                "num_results": 3,
            },
        )
        finder_results = _first_result_item(finder_informal)
        if finder_results:
            assert isinstance(finder_results, dict) and len(finder_results.keys()) == 3
            assert {"full_name", "formal_statement", "informal_statement"} <= set(
                finder_results.keys()
            )
        else:
            finder_text = result_text(finder_informal)
            assert finder_text and len(finder_text) > 0
