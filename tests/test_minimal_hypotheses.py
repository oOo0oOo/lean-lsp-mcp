"""Integration tests for lean_minimal_hypotheses tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.mark.asyncio
@pytest.mark.slow
async def test_minimal_hypotheses_one_unused(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """h1 is referenced in the body, h2 is not — so h2 is removable, h1 load-bearing.
    The load-bearing verdict's `breaks` should pinpoint the body line that names h1."""
    test_file = test_project_path / "MinimalHypothesesTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_minimal_hypotheses",
            {
                "file_path": str(test_file),
                "theorem_name": "minhyp_one_unused",
            },
        )
        data = result_json(result)
        by_binder = {v["binder"]: v for v in data["verdicts"]}
        assert by_binder["(h1 : 1 + 1 = 2)"]["status"] == "load-bearing"
        assert by_binder["(h2 : 2 + 2 = 4)"]["status"] == "removable"

        # Removing h1 must pinpoint the body line that names h1.
        h1_breaks = by_binder["(h1 : 1 + 1 = 2)"]["breaks"]
        assert h1_breaks, "expected `breaks` to enumerate the new errors"
        assert any("h1" in b["message"] for b in h1_breaks), (
            f"expected an error referencing h1, got: {h1_breaks}"
        )
        # Removing the unused h2 must produce no breaks.
        assert by_binder["(h2 : 2 + 2 = 4)"]["breaks"] == []
        assert data["skipped_implicit"] == 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_minimal_hypotheses_both_used(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    test_file = test_project_path / "MinimalHypothesesTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_minimal_hypotheses",
            {
                "file_path": str(test_file),
                "theorem_name": "minhyp_both_used",
            },
        )
        data = result_json(result)
        statuses = [v["status"] for v in data["verdicts"]]
        assert statuses == ["load-bearing", "load-bearing"]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_minimal_hypotheses_no_explicit(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    test_file = test_project_path / "MinimalHypothesesTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_minimal_hypotheses",
            {
                "file_path": str(test_file),
                "theorem_name": "minhyp_no_hypotheses",
            },
        )
        data = result_json(result)
        assert data["verdicts"] == []
        assert data["skipped_implicit"] == 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_minimal_hypotheses_skips_implicit_and_instance(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """`{α : Type}` and `[DecidableEq α]` are skipped — only h1/h2 are probed."""
    test_file = test_project_path / "MinimalHypothesesTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_minimal_hypotheses",
            {
                "file_path": str(test_file),
                "theorem_name": "minhyp_mixed_binders",
            },
        )
        data = result_json(result)
        verdicts = {v["binder"]: v["status"] for v in data["verdicts"]}
        assert verdicts == {
            "(h1 : 1 + 1 = 2)": "load-bearing",
            "(h2 : 2 + 2 = 4)": "removable",
        }
        # {α : Type} and [DecidableEq α]
        assert data["skipped_implicit"] == 2


@pytest.mark.asyncio
async def test_minimal_hypotheses_unknown_theorem(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    test_file = test_project_path / "MinimalHypothesesTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_minimal_hypotheses",
            {
                "file_path": str(test_file),
                "theorem_name": "nonexistent_xyz",
            },
            expect_error=True,
        )
        assert result.isError
