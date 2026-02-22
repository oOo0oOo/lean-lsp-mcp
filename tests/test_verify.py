"""Integration tests for lean_verify tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.mark.asyncio
async def test_verify_clean_theorem(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_verify",
            {
                "file_path": str(verify_file),
                "theorem_name": "verify_clean",
            },
        )
        data = result_json(result)
        # Clean theorem should only have standard axioms (or none)
        standard = {"propext", "Classical.choice", "Quot.sound"}
        for ax in data["axioms"]:
            assert ax in standard, f"Unexpected axiom: {ax}"


@pytest.mark.asyncio
async def test_verify_sorry_theorem(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_verify",
            {
                "file_path": str(verify_file),
                "theorem_name": "verify_sorry",
            },
        )
        data = result_json(result)
        assert "sorryAx" in data["axioms"]


@pytest.mark.asyncio
async def test_verify_warnings(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_verify",
            {
                "file_path": str(verify_file),
                "theorem_name": "verify_clean",
                "scan_source": True,
            },
        )
        data = result_json(result)
        patterns = [w["pattern"] for w in data["warnings"]]
        assert any("debug." in p for p in patterns)
        assert any("unsafe" in p for p in patterns)


@pytest.mark.asyncio
async def test_verify_no_warnings(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_verify",
            {
                "file_path": str(verify_file),
                "theorem_name": "verify_clean",
                "scan_source": False,
            },
        )
        data = result_json(result)
        assert data["warnings"] == []


@pytest.mark.asyncio
async def test_verify_nonexistent_theorem(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    verify_file = test_project_path / "VerifyTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_verify",
            {
                "file_path": str(verify_file),
                "theorem_name": "nonexistent_theorem_xyz",
            },
            expect_error=True,
        )
        assert result.isError
