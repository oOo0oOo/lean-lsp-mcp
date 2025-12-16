from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, MCPToolError, result_text


def parse_diagnostics_result(result) -> list[dict]:
    """Parse diagnostics result, handling both structured and text formats."""
    if result.structuredContent is not None:
        return result.structuredContent.get("items", [])
    # Fallback to parsing text output
    text = result_text(result).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
        return parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        return []


@pytest.fixture(scope="module")
def diagnostic_file(test_project_path: Path) -> Path:
    path = test_project_path / "DiagnosticTest.lean"
    content = textwrap.dedent(
        """
        import Mathlib

        -- Line 3: Valid definition
        def validDef : Nat := 42

        -- Line 6: Error on this line
        def errorDef : Nat := "string"

        -- Line 9: Another valid definition
        def anotherValidDef : Nat := 100

        -- Line 12: Another error
        def anotherError : String := 123

        -- Line 15: Valid theorem
        theorem validTheorem : True := by
          trivial
        """
    ).strip()
    if not path.exists() or path.read_text(encoding="utf-8") != content + "\n":
        path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_diagnostic_messages_line_filtering(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    diagnostic_file: Path,
) -> None:
    """Test all line range filtering scenarios in one client session."""
    async with mcp_client_factory() as client:
        # Test 1: Get all diagnostic messages without line range filtering
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(diagnostic_file)},
        )
        all_items = parse_diagnostics_result(diagnostics)
        # Should contain at least 2 errors
        assert len(all_items) >= 2, (
            f"Expected at least 2 diagnostics, got {len(all_items)}"
        )
        # Verify items have expected structure
        for item in all_items:
            assert "severity" in item
            assert "message" in item
            assert "line" in item

        # Test 2: Get diagnostics starting from line 10
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
                "start_line": 10,
            },
        )
        filtered_items = parse_diagnostics_result(diagnostics)
        # Should have fewer diagnostics than unfiltered
        assert len(filtered_items) < len(all_items)

        # Test 3: Get diagnostics for specific line range
        if all_items:
            first_error_line = all_items[0]["line"]
            diagnostics = await client.call_tool(
                "lean_diagnostic_messages",
                {
                    "file_path": str(diagnostic_file),
                    "start_line": 1,
                    "end_line": first_error_line,
                },
            )
            range_items = parse_diagnostics_result(diagnostics)
            assert len(range_items) >= 1
            assert len(range_items) < len(all_items)

        # Test 4: Get diagnostics for range with no errors (lines 14-17)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
                "start_line": 14,
                "end_line": 17,
            },
        )
        empty_items = parse_diagnostics_result(diagnostics)
        # Should be empty
        assert len(empty_items) == 0, f"Expected no diagnostics, got {len(empty_items)}"


@pytest.fixture(scope="module")
def declaration_diagnostic_file(test_project_path: Path) -> Path:
    """Create a test file with multiple declarations, some with errors."""
    path = test_project_path / "DeclarationDiagnosticTest.lean"
    content = textwrap.dedent(
        """
        import Mathlib

        -- First theorem with a clear type error
        theorem firstTheorem : 1 + 1 = 2 := "string instead of proof"

        -- Valid definition
        def validFunction : Nat := 42

        -- Second theorem with an error in the statement type mismatch
        theorem secondTheorem : Nat := True

        -- Another valid definition
        def anotherValidFunction : String := "hello"
        """
    ).strip()
    if not path.exists() or path.read_text(encoding="utf-8") != content + "\n":
        path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_diagnostic_messages_declaration_filtering(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    declaration_diagnostic_file: Path,
) -> None:
    """Test all declaration-based filtering scenarios in one client session."""
    async with mcp_client_factory() as client:
        # Test 1: Get all diagnostics first to verify file has errors
        all_diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(declaration_diagnostic_file)},
        )
        all_items = parse_diagnostics_result(all_diagnostics)
        assert len(all_items) > 0, "Expected diagnostics in file with errors"

        # Test 2: Get diagnostics for firstTheorem only
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "firstTheorem",
            },
        )
        first_items = parse_diagnostics_result(diagnostics)
        assert len(first_items) > 0
        assert len(first_items) <= len(all_items)

        # Test 3: Get diagnostics for secondTheorem (has type error in statement)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "secondTheorem",
            },
        )
        second_items = parse_diagnostics_result(diagnostics)
        assert len(second_items) > 0

        # Test 4: Get diagnostics for validFunction (no errors)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "validFunction",
            },
        )
        valid_items = parse_diagnostics_result(diagnostics)
        assert len(valid_items) == 0, (
            f"Expected no diagnostics for valid function, got {len(valid_items)}"
        )


@pytest.mark.asyncio
async def test_diagnostic_messages_declaration_edge_cases(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    declaration_diagnostic_file: Path,
) -> None:
    """Test edge cases for declaration-based filtering."""
    async with mcp_client_factory() as client:
        # Test 1: Non-existent declaration - now raises MCPToolError
        with pytest.raises(MCPToolError) as exc_info:
            await client.call_tool(
                "lean_diagnostic_messages",
                {
                    "file_path": str(declaration_diagnostic_file),
                    "declaration_name": "nonExistentTheorem",
                },
            )
        assert "not found" in str(exc_info.value).lower()

        # Test 2: declaration_name takes precedence over start_line/end_line
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "firstTheorem",
                "start_line": 1,  # These should be ignored
                "end_line": 3,  # These should be ignored
            },
        )
        diag_text = result_text(diagnostics)
        # Should get diagnostics for firstTheorem, not lines 1-3
        assert len(diag_text) > 0


@pytest.fixture(scope="module")
def kernel_error_file(test_project_path: Path) -> Path:
    """File with kernel error as first error (issue #63)."""
    path = test_project_path / "KernelErrorTest.lean"
    content = textwrap.dedent(
        """
        import Mathlib.Data.Real.Basic

        structure test where
          x : ℝ
          deriving Repr

        lemma test_lemma : False := by rfl
        """
    ).strip()
    if not path.exists() or path.read_text(encoding="utf-8") != content + "\n":
        path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_diagnostic_messages_detects_kernel_errors(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    kernel_error_file: Path,
) -> None:
    """Test kernel errors detected when first in file (issue #63)."""
    async with mcp_client_factory() as client:
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(kernel_error_file)},
        )
        items = parse_diagnostics_result(diagnostics)

        # Should have at least 2 diagnostics
        assert len(items) >= 2, f"Expected at least 2 diagnostics, got {len(items)}"

        # Check for kernel error and regular error in message content
        all_messages = " ".join(item.get("message", "").lower() for item in items)
        assert "kernel" in all_messages or "unsafe" in all_messages
        assert "rfl" in all_messages or "failed" in all_messages
