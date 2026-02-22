from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture(scope="module")
def code_actions_file(test_project_path: Path) -> Path:
    path = test_project_path / "CodeActionsTest.lean"
    content = "import Mathlib\n\nexample (n : Nat) : 0 + n = n := by simp?\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_code_actions_try_this(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    code_actions_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_code_actions",
            {"file_path": str(code_actions_file), "line": 3},
        )
        text = result_text(result)
        data = json.loads(text)

        assert "actions" in data
        actions = data["actions"]
        assert len(actions) >= 1

        # At least one action should be a TryThis suggestion
        try_this = [a for a in actions if "Try this" in a["title"]]
        assert len(try_this) >= 1

        action = try_this[0]
        assert action["is_preferred"] is True or action["is_preferred"] is False
        assert len(action["edits"]) >= 1

        edit = action["edits"][0]
        assert edit["start_line"] == 3
        assert "new_text" in edit
        assert edit["new_text"]  # non-empty replacement


@pytest.mark.asyncio
async def test_code_actions_empty_on_clean_line(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    code_actions_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_code_actions",
            {"file_path": str(code_actions_file), "line": 1},
        )
        data = json.loads(result_text(result))
        assert data["actions"] == []
