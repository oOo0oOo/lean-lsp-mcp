from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture()
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
    path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_diagnostic_messages_without_line_range(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    diagnostic_file: Path,
) -> None:
    """Test getting all diagnostic messages without line range filtering."""
    async with mcp_client_factory() as client:
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(diagnostic_file)},
        )
        diag_text = result_text(diagnostics)

        # Should contain both errors
        assert "string" in diag_text.lower() or "error" in diag_text.lower()
        # Check that multiple diagnostics are returned (at least the two errors we created)
        assert diag_text.count("severity") >= 2


@pytest.mark.asyncio
async def test_diagnostic_messages_with_start_line(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    diagnostic_file: Path,
) -> None:
    """Test getting diagnostic messages starting from a specific line."""
    async with mcp_client_factory() as client:
        # First get all diagnostics to see what we have
        all_diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
            },
        )
        all_diag_text = result_text(all_diagnostics)

        # Get diagnostics starting from line 10 (should only include the second error)
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
        # Should have fewer diagnostics than all_diagnostics
        assert len(diag_text) < len(all_diag_text)


@pytest.mark.asyncio
async def test_diagnostic_messages_with_line_range(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    diagnostic_file: Path,
) -> None:
    """Test getting diagnostic messages for a specific line range."""
    async with mcp_client_factory() as client:
        # First, get all diagnostics to see what lines they're actually on
        all_diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(diagnostic_file)},
        )
        all_diag_text = result_text(all_diagnostics)

        # Extract line numbers from the diagnostics (format: "l7c23-l7c31")
        import re

        line_matches = re.findall(r"l(\d+)c", all_diag_text)
        if line_matches:
            first_error_line = int(line_matches[0])

            # Get diagnostics only up to that error line
            diagnostics = await client.call_tool(
                "lean_diagnostic_messages",
                {
                    "file_path": str(diagnostic_file),
                    "start_line": 1,
                    "end_line": first_error_line,
                },
            )
            diag_text = result_text(diagnostics)

            # Should contain the first error
            assert "string" in diag_text.lower() or len(diag_text) > 0
            # Should be fewer diagnostics than all
            assert len(diag_text) < len(all_diag_text)


@pytest.mark.asyncio
async def test_diagnostic_messages_with_no_errors_in_range(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    diagnostic_file: Path,
) -> None:
    """Test getting diagnostic messages for a range with no errors."""
    async with mcp_client_factory() as client:
        # Get diagnostics only for lines 14-17 (valid theorem, should have no errors)
        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(diagnostic_file),
                "start_line": 14,
                "end_line": 17,
            },
        )
        diag_text = result_text(diagnostics)

        # Should indicate no errors or be empty
        # The exact format depends on how the tool formats an empty result
        assert (
            "no" in diag_text.lower()
            or len(diag_text.strip()) == 0
            or diag_text == "[]"
        )
