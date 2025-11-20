"""Test outline generation with various Lean files."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture
def mathlib_nat_basic(test_project_path: Path) -> Path:
    """Path to Mathlib Data.Nat.Basic file."""
    return test_project_path / ".lake/packages/mathlib/Mathlib/Data/Nat/Basic.lean"


@pytest.mark.asyncio
async def test_outline_simple_files(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """Test outline generation on simple test files."""
    test_files = [
        test_project_path / "StructTest.lean",
        test_project_path / "TheoremTest.lean",
    ]
    
    async with mcp_client_factory() as client:
        for test_file in test_files:
            result = await client.call_tool("lean_file_outline", {"file_path": str(test_file)})
            outline = result_text(result)
            
            # Basic structure checks
            assert "## Imports" in outline or "## Declarations" in outline
            assert len(outline) > 0


@pytest.mark.asyncio
async def test_mathlib_outline_structure(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Test outline generation with a real Mathlib file."""
    async with mcp_client_factory() as client:
        result = await client.call_tool("lean_file_outline", {"file_path": str(mathlib_nat_basic)})
        outline = result_text(result)
        
        # Basic structure checks (no filename header now)
        assert "## Imports" in outline
        assert "## Declarations" in outline
        
        # Should have imports from Mathlib
        assert "Mathlib.Data.Nat.Init" in outline
        
        # Should have namespace (new format)
        assert "[Ns:" in outline and "Nat" in outline
        
        # Should have instance declarations
        assert "instLinearOrder" in outline or "LinearOrder" in outline


@pytest.mark.asyncio
async def test_mathlib_outline_has_line_numbers(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify line numbers are present in outline."""
    async with mcp_client_factory() as client:
        result = await client.call_tool("lean_file_outline", {"file_path": str(mathlib_nat_basic)})
        outline = result_text(result)
        
        # Should have line numbers in format "[Tag: L27-135]" or "[Tag: L31]"
        line_pattern = r'L(\d+)(?:-(\d+))?'
        matches = re.findall(line_pattern, outline)
        
        assert len(matches) > 0, "Should have line number annotations"


@pytest.mark.asyncio
async def test_mathlib_outline_has_types(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify type signatures are included."""
    async with mcp_client_factory() as client:
        result = await client.call_tool("lean_file_outline", {"file_path": str(mathlib_nat_basic)})
        outline = result_text(result)

        # Should have type annotations with ":"
        assert "LinearOrder ℕ" in outline or ": " in outline


@pytest.mark.asyncio
async def test_mathlib_outline_file_cleanup(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify file is properly cleaned up after info_trees extraction."""
    async with mcp_client_factory() as client:
        # Get original file content
        original_content = mathlib_nat_basic.read_text()

        # Generate outline (which inserts and removes #info_trees lines)
        await client.call_tool("lean_file_outline", {"file_path": str(mathlib_nat_basic)})

        # Read file content again
        final_content = mathlib_nat_basic.read_text()

        # File should be unchanged
        assert final_content == original_content, "File should be restored to original state after outline generation"

        # Specifically check that no #info_trees lines remain
        assert "#info_trees" not in final_content, "No #info_trees directives should remain in file"
