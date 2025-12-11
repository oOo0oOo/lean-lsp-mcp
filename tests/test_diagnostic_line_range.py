from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, MCPToolError, result_text


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
        diag_text = result_text(diagnostics)
        # Should contain both errors - now returns JSON with "severity" field
        assert "string" in diag_text.lower() or "error" in diag_text.lower()
        # Count occurrences of "severity" in JSON output (appears as field name)
        assert diag_text.count('"severity"') >= 2
        all_diag_text = diag_text

        # Test 2: Get diagnostics starting from line 10
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
                "start_line": 10,
            },
        )
        diag_text = result_text(diagnostics)
        # Should contain the second error (line 13: anotherError)
        assert "123" in diag_text or "error" in diag_text.lower()
        assert len(diag_text) < len(all_diag_text)

        # Test 3: Get diagnostics for specific line range
        import re

        # Extract start_line from JSON format (e.g., "start_line": 7)
        line_matches = re.findall(r'"start_line":\s*(\d+)', all_diag_text)
        if line_matches:
            first_error_line = int(line_matches[0])
            diagnostics = await client.call_tool(
                "lean_diagnostic_messages",
                {
                    "file_path": str(diagnostic_file),
                    "start_line": 1,
                    "end_line": first_error_line,
                },
            )
            diag_text = result_text(diagnostics)
            assert "string" in diag_text.lower() or len(diag_text) > 0
            assert len(diag_text) < len(all_diag_text)

        # Test 4: Get diagnostics for range with no errors (lines 14-17)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
                "start_line": 14,
                "end_line": 17,
            },
        )
        diag_text = result_text(diagnostics)
        # Empty array or no diagnostics
        assert diag_text.strip() == "[]" or len(diag_text.strip()) == 0


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
        all_diag_text = result_text(all_diagnostics)
        assert len(all_diag_text) > 0
        assert "string" in all_diag_text.lower() or "type" in all_diag_text.lower()

        # Test 2: Get diagnostics for firstTheorem only
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "firstTheorem",
            },
        )
        diag_text = result_text(diagnostics)
        assert len(diag_text) > 0
        assert len(diag_text) <= len(all_diag_text)

        # Test 3: Get diagnostics for secondTheorem (has type error in statement)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "secondTheorem",
            },
        )
        diag_text = result_text(diagnostics)
        assert len(diag_text) > 0
        assert isinstance(diag_text, str)

        # Test 4: Get diagnostics for validFunction (no errors)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(declaration_diagnostic_file),
                "declaration_name": "validFunction",
            },
        )
        diag_text = result_text(diagnostics)
        assert (
            "no" in diag_text.lower()
            or len(diag_text.strip()) == 0
            or diag_text == "[]"
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
        diag_text = result_text(diagnostics)

        # Detect kernel error and regular error
        assert "kernel" in diag_text.lower() or "unsafe" in diag_text.lower()
        assert "rfl" in diag_text.lower() or "failed" in diag_text.lower()
        assert diag_text.count("severity") >= 2
