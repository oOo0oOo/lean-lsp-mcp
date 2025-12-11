"""Test outline generation with various Lean files."""

from __future__ import annotations

import orjson
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


def _parse_outline(result) -> dict:
    """Parse the JSON outline result."""
    text = result_text(result)
    return orjson.loads(text)


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

    # Skip if test files don't exist
    existing_files = [f for f in test_files if f.exists()]
    if not existing_files:
        pytest.skip("Test files StructTest.lean/TheoremTest.lean not found")

    async with mcp_client_factory() as client:
        for test_file in existing_files:
            result = await client.call_tool(
                "lean_file_outline", {"file_path": str(test_file)}
            )
            outline = _parse_outline(result)

            # Basic structure checks - now JSON with imports and declarations
            assert "imports" in outline or "declarations" in outline
            assert isinstance(outline.get("imports", []), list)
            assert isinstance(outline.get("declarations", []), list)


@pytest.mark.asyncio
async def test_mathlib_outline_structure(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Test outline generation with a real Mathlib file."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # Basic structure checks - now structured JSON
        assert "imports" in outline
        assert "declarations" in outline

        # Should have imports from Mathlib
        assert any("Mathlib.Data.Nat.Init" in imp for imp in outline["imports"])

        # Should have namespace declarations with kind "Ns"
        decl_names = [d["name"] for d in outline["declarations"]]
        decl_kinds = [d["kind"] for d in outline["declarations"]]
        assert "Nat" in decl_names or any("Ns" in k for k in decl_kinds)

        # Should have instance declarations
        assert any(
            "instLinearOrder" in d["name"]
            or "LinearOrder" in (d.get("type_signature") or "")
            for d in outline["declarations"]
        )


@pytest.mark.asyncio
async def test_mathlib_outline_has_line_numbers(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify line numbers are present in outline."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # All declarations should have start_line and end_line
        for decl in outline["declarations"]:
            assert "start_line" in decl and isinstance(decl["start_line"], int)
            assert "end_line" in decl and isinstance(decl["end_line"], int)
            assert decl["start_line"] > 0
            assert decl["end_line"] >= decl["start_line"]


@pytest.mark.asyncio
async def test_mathlib_outline_has_types(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    mathlib_nat_basic: Path,
) -> None:
    """Verify type signatures are included."""
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )
        outline = _parse_outline(result)

        # Some declarations should have type_signature
        has_type_sig = any(d.get("type_signature") for d in outline["declarations"])
        assert has_type_sig, "At least some declarations should have type signatures"


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
        await client.call_tool(
            "lean_file_outline", {"file_path": str(mathlib_nat_basic)}
        )

        # Read file content again
        final_content = mathlib_nat_basic.read_text()

        # File should be unchanged
        assert final_content == original_content, (
            "File should be restored to original state after outline generation"
        )

        # Specifically check that no #info_trees lines remain
        assert "#info_trees" not in final_content, (
            "No #info_trees directives should remain in file"
        )
