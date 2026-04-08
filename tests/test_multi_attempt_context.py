from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.fixture()
def multi_attempt_context_file(test_project_path: Path) -> Path:
    return test_project_path / "MultiAttemptContextTest.lean"


def _assert_no_command_parser_errors(data: dict) -> None:
    parser_errors = [
        diag["message"]
        for item in data["items"]
        for diag in item.get("diagnostics", [])
        if "expected command" in diag["message"]
    ]
    assert not parser_errors


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


@pytest.mark.asyncio
async def test_multi_attempt_later_tactic_lines_use_tactic_context(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    multi_attempt_context_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        expected_attempts = {
            5: {
                "goals_before": ["case left\n⊢ True", "case right\n⊢ True"],
                "trivial_goals": ["case right\n⊢ True"],
            },
            6: {
                "goals_before": ["case right\n⊢ True"],
                "trivial_goals": [],
            },
        }

        for line, expected in expected_attempts.items():
            goal = await client.call_tool(
                "lean_goal",
                {
                    "file_path": str(multi_attempt_context_file),
                    "line": line,
                },
            )
            goal_data = result_json(goal)
            assert goal_data["goals_before"] == expected["goals_before"]

            result = await client.call_tool(
                "lean_multi_attempt",
                {
                    "file_path": str(multi_attempt_context_file),
                    "line": line,
                    "snippets": ["trivial", "exact True.intro", "rfl"],
                },
            )
            data = result_json(result)
            _assert_no_command_parser_errors(data)

            trivial_attempt = data["items"][0]
            assert trivial_attempt["snippet"] == "trivial"
            assert trivial_attempt["goals"] == expected["trivial_goals"]
            assert trivial_attempt["diagnostics"] == []


@pytest.mark.asyncio
async def test_multi_attempt_column_aligns_with_goal_position(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    multi_attempt_context_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        goal = await client.call_tool(
            "lean_goal",
            {
                "file_path": str(multi_attempt_context_file),
                "line": 5,
                "column": 3,
            },
        )
        goal_data = result_json(goal)
        assert goal_data["goals"] == [
            "case left\n⊢ True",
            "case right\n⊢ True",
        ]

        result = await client.call_tool(
            "lean_multi_attempt",
            {
                "file_path": str(multi_attempt_context_file),
                "line": 5,
                "column": 3,
                "snippets": ["trivial", "exact True.intro", "rfl"],
            },
        )
        data = result_json(result)
        _assert_no_command_parser_errors(data)

        trivial_attempt = data["items"][0]
        assert trivial_attempt["snippet"] == "trivial"
        assert trivial_attempt["goals"] == ["case right\n⊢ True"]
        assert trivial_attempt["diagnostics"] == []
