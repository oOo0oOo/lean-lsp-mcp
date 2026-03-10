"""Integration tests for lean_goal_tracker tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.mark.asyncio
async def test_goal_tracker_clean(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_goal_tracker",
            {
                "file_path": str(verify_file),
                "decl_name": "verify_clean",
            },
        )
        data = result_json(result)
        assert data["target"] == "verify_clean"
        assert len(data["sorry_declarations"]) == 0
        assert data["total_transitive_deps"] > 0


@pytest.mark.asyncio
async def test_goal_tracker_sorry(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_goal_tracker",
            {
                "file_path": str(verify_file),
                "decl_name": "verify_sorry",
            },
        )
        data = result_json(result)
        assert data["target"] == "verify_sorry"
        assert len(data["sorry_declarations"]) >= 1
        sorry_names = [s["name"] for s in data["sorry_declarations"]]
        assert "verify_sorry" in sorry_names
        assert data["total_transitive_deps"] > 0


@pytest.mark.asyncio
async def test_goal_tracker_nonexistent(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_goal_tracker",
            {
                "file_path": str(verify_file),
                "decl_name": "nonexistent_decl_xyz",
            },
            expect_error=True,
        )
        assert result.isError
