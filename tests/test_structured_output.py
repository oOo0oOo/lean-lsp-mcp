"""Test that tool outputs are properly structured JSON, not double-encoded strings.

This test addresses an issue where tools returning lists would produce
double-encoded JSON in the structuredContent field. For example, instead of:

    {"items": [{"severity": "error", ...}]}

The output would be:

    {"result": "[\\n  {\\\"severity\\\": \\\"error\\\", ...}]"}

Where the list is encoded as an escaped string instead of a proper JSON array.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


@pytest.fixture(scope="module")
def error_file(test_project_path: Path) -> Path:
    """Create a Lean file with a type error for testing diagnostics."""
    path = test_project_path / "StructuredOutputTest.lean"
    content = textwrap.dedent(
        """
        import Mathlib

        -- This line has a type error: assigning String to Nat
        def badDef : Nat := "not a number"
        """
    ).strip()
    if not path.exists() or path.read_text(encoding="utf-8") != content + "\n":
        path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_diagnostics_structured_output_not_double_encoded(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    error_file: Path,
) -> None:
    """Verify diagnostics are returned as structured data, not double-encoded JSON.

    The structuredContent should contain an 'items' field with a list of
    diagnostic objects, not a 'result' field with an escaped JSON string.
    """
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(error_file)},
        )

        # structuredContent should be present
        assert result.structuredContent is not None, (
            "Tool should return structured content"
        )

        structured = result.structuredContent

        # The result should have an 'items' field, not a 'result' field
        # If it has 'result' with a string value, that indicates double-encoding
        if "result" in structured:
            result_value = structured["result"]
            if isinstance(result_value, str):
                # Try to parse it as JSON to confirm it's double-encoded
                try:
                    parsed = json.loads(result_value)
                    pytest.fail(
                        f"Diagnostics are double-encoded! "
                        f"structuredContent['result'] is a JSON string that parses to: {type(parsed).__name__}. "
                        f"Expected structuredContent to contain 'items' with a list directly."
                    )
                except json.JSONDecodeError:
                    pass  # Not JSON, different issue

        # Should have 'items' field with a list
        assert "items" in structured, (
            f"Expected 'items' field in structuredContent, got keys: {list(structured.keys())}"
        )
        items = structured["items"]
        assert isinstance(items, list), (
            f"Expected 'items' to be a list, got {type(items).__name__}"
        )

        # Each item should be a dict with proper fields, not strings
        for i, item in enumerate(items):
            assert isinstance(item, dict), (
                f"Item {i} should be a dict, got {type(item).__name__}. "
                f"This suggests the list items are double-encoded as strings."
            )

            # Verify the diagnostic has the expected fields as proper types
            assert "severity" in item, f"Item {i} missing 'severity' field"
            assert isinstance(item["severity"], str), (
                f"Item {i} 'severity' should be a string, got {type(item['severity']).__name__}"
            )

            assert "message" in item, f"Item {i} missing 'message' field"
            assert isinstance(item["message"], str), (
                f"Item {i} 'message' should be a string, got {type(item['message']).__name__}"
            )

            assert "line" in item, f"Item {i} missing 'line' field"
            assert isinstance(item["line"], int), (
                f"Item {i} 'line' should be an int, got {type(item['line']).__name__}"
            )

            assert "column" in item, f"Item {i} missing 'column' field"
            assert isinstance(item["column"], int), (
                f"Item {i} 'column' should be an int, got {type(item['column']).__name__}"
            )

        # We should have at least one diagnostic (the type error)
        assert len(items) >= 1, (
            "Expected at least one diagnostic for the type error in the test file"
        )
