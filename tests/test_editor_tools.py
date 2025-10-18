from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.fixture()
def editor_file(test_project_path: Path) -> Path:
    path = test_project_path / "EditorTools.lean"
    content = (
        "\n".join(
            [
                "import Mathlib",
                "",
                "def sampleValue : Nat := 42",
                "",
                "theorem sampleTheorem : True := by",
                "  trivial",
                "",
                "def completionTest : Nat := Nat.su",
            ]
        )
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_editor_tools(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    editor_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        contents = await client.call_tool(
            "lean_file_contents",
            {
                "file_path": str(editor_file),
                "annotate_lines": False,
            },
        )
        text = result_text(contents)
        assert "sampleValue" in text
        lines = text.splitlines()

        diagnostics = await client.call_tool(
            "lean_diagnostic_messages",
            {"file_path": str(editor_file)},
        )
        diag_text = result_text(diagnostics)
        assert "Nat.su" in diag_text

        goal = await client.call_tool(
            "lean_goal",
            {
                "file_path": str(editor_file),
                "line": 6,
                "column": lines[5].index("trivial") + 1,
            },
        )
        assert "⊢ True" in result_text(goal)

        completion_line = lines[7]
        term_goal = await client.call_tool(
            "lean_term_goal",
            {
                "file_path": str(editor_file),
                "line": 8,
                "column": completion_line.index("Nat.su") + 1,
            },
        )
        assert "⊢ ℕ" in result_text(term_goal)

        hover = await client.call_tool(
            "lean_hover_info",
            {
                "file_path": str(editor_file),
                "line": 3,
                "column": lines[2].index("sampleValue") + 1,
            },
        )
        hover_text = result_text(hover)
        assert "sampleValue" in hover_text

        column = completion_line.index("Nat.su") + len("Nat.su")
        completions = await client.call_tool(
            "lean_completions",
            {
                "file_path": str(editor_file),
                "line": 8,
                "column": column + 1,
                "max_completions": 500,
            },
        )
        assert "succ" in result_text(completions)
