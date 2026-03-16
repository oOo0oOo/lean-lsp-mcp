from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.fixture()
def multi_attempt_context_file(test_project_path: Path) -> Path:
    return test_project_path / "MultiAttemptContextTest.lean"


@pytest.mark.asyncio
async def test_multi_attempt_first_tactic_line_works(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    multi_attempt_context_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        goal = await client.call_tool(
            "lean_goal",
            {
                "file_path": str(multi_attempt_context_file),
                "line": 4,
            },
        )
        goal_data = result_json(goal)
        assert goal_data["goals_before"] == ["⊢ True ∧ True"]
        assert goal_data["goals_after"] == [
            "case left\n⊢ True",
            "case right\n⊢ True",
        ]

        result = await client.call_tool(
            "lean_multi_attempt",
            {
                "file_path": str(multi_attempt_context_file),
                "line": 4,
                "snippets": ["constructor", "trivial", "rfl"],
            },
        )
        data = result_json(result)
        constructor_attempt = data["items"][0]

        assert constructor_attempt["snippet"] == "constructor"
        assert constructor_attempt["goals"] == [
            "case left\n⊢ True",
            "case right\n⊢ True",
        ]
        assert constructor_attempt["diagnostics"] == []


@pytest.mark.xfail(
    strict=True,
    reason="lean_multi_attempt loses tactic context after the first proof line",
)
@pytest.mark.asyncio
async def test_multi_attempt_later_tactic_lines_use_tactic_context(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    multi_attempt_context_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        expected_goals_before = {
            5: ["case left\n⊢ True", "case right\n⊢ True"],
            6: ["case right\n⊢ True"],
        }

        for line, goals_before in expected_goals_before.items():
            goal = await client.call_tool(
                "lean_goal",
                {
                    "file_path": str(multi_attempt_context_file),
                    "line": line,
                },
            )
            goal_data = result_json(goal)
            assert goal_data["goals_before"] == goals_before

            result = await client.call_tool(
                "lean_multi_attempt",
                {
                    "file_path": str(multi_attempt_context_file),
                    "line": line,
                    "snippets": ["trivial", "exact True.intro", "rfl"],
                },
            )
            data = result_json(result)

            parser_errors = [
                diag["message"]
                for item in data["items"]
                for diag in item.get("diagnostics", [])
                if "expected command" in diag["message"]
            ]
            assert not parser_errors

            trivial_attempt = data["items"][0]
            assert trivial_attempt["snippet"] == "trivial"
            assert trivial_attempt["goals"] == []
            assert trivial_attempt["diagnostics"] == []
