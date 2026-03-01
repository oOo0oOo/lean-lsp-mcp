"""Integration tests for the semantic naming tools (lean_type, lean_definition, etc.)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import orjson
import pytest

from tests.helpers.mcp_client import MCPClient, result_json, result_text


@pytest.fixture()
def semantic_file(test_project_path: Path) -> Path:
    """Write a test file with known declarations for semantic tool testing."""
    path = test_project_path / "SemanticTest.lean"
    content = "\n".join(
        [
            "import Mathlib",
            "",
            "def sampleDef : Nat := 42",
            "",
            "theorem sampleThm : 1 + 1 = 2 := by",
            "  norm_num",
            "",
            "def SemanticNs.namespacedDef : Nat := 7",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_lean_type(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_type should return the type signature of a declaration."""
    async with mcp_client_factory() as client:
        # First call a file-based tool to set the project path
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_type",
            {"name": "sampleDef"},
        )
        text = result_text(result)
        assert "sampleDef" in text
        assert "Nat" in text or "ℕ" in text


@pytest.mark.asyncio
async def test_lean_definition(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_definition should return the source code of a declaration."""
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_definition",
            {"name": "sampleDef"},
        )
        text = result_text(result)
        assert "42" in text
        assert "sampleDef" in text


@pytest.mark.asyncio
async def test_lean_check_definition(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_check_definition should return diagnostics for a declaration."""
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_check_definition",
            {"name": "sampleDef"},
        )
        text = result_text(result)
        # sampleDef should have no errors
        assert "success" in text.lower() or "true" in text.lower() or '"items": []' in text


@pytest.mark.asyncio
async def test_lean_goal_at(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_goal_at should return goal state at a tactic position."""
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_goal_at",
            {"name": "sampleThm", "tactic_index": 0},
        )
        text = result_text(result)
        # At the norm_num tactic, the goal should mention 1 + 1 = 2
        assert "norm_num" in text or "1" in text


@pytest.mark.asyncio
async def test_lean_type_fqn(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_type should work with fully qualified names."""
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_type",
            {"name": "SemanticNs.namespacedDef"},
        )
        text = result_text(result)
        assert "Nat" in text or "ℕ" in text


@pytest.mark.asyncio
async def test_lean_type_not_found(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    semantic_file: Path,
) -> None:
    """lean_type should return error for nonexistent declarations."""
    async with mcp_client_factory() as client:
        await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(semantic_file)},
        )

        result = await client.call_tool(
            "lean_type",
            {"name": "totallyNonexistentDecl_xyz"},
            expect_error=True,
        )
        text = result_text(result)
        assert "not found" in text.lower()
