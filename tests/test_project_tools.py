from __future__ import annotations

from pathlib import Path

import pytest

from collections.abc import Callable
from typing import AsyncContextManager

from tests.helpers.mcp_client import MCPClient, result_text


def _mathlib_file(test_project_path: Path) -> Path:
    candidate = (
        test_project_path
        / ".lake"
        / "packages"
        / "mathlib"
        / "Mathlib"
        / "Data"
        / "Nat"
        / "Basic.lean"
    )
    if not candidate.exists():
        pytest.skip(
            "mathlib sources were not downloaded; run `lake update` inside tests/test_project."
        )
    return candidate


def _first_occurrence_location(source: str, needle: str) -> tuple[int, int]:
    for line_number, line in enumerate(source.splitlines(), start=1):
        if needle not in line:
            continue
        stripped = line.lstrip()
        if (
            stripped.startswith("--")
            or stripped.startswith("/-")
            or stripped.startswith("*")
        ):
            continue
        column = line.index(needle) + 1
        return line_number, column

    pytest.skip(
        f"Cannot find `{needle}` in executable mathlib code; update the test to match new contents."
    )


@pytest.mark.asyncio
async def test_mathlib_file_roundtrip(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    target_file = _mathlib_file(test_project_path)

    async with mcp_client_factory() as mcp_client:
        contents = await mcp_client.call_tool(
            "lean_file_contents",
            {
                "file_path": str(target_file),
                "annotate_lines": False,
            },
        )
        text = result_text(contents)
        assert "import Mathlib" in text

        line, column = _first_occurrence_location(text, "Nat.succ")

        hover = await mcp_client.call_tool(
            "lean_hover_info",
            {
                "file_path": str(target_file),
                "line": line,
                "column": column,
            },
        )
        hover_text = result_text(hover)
        assert "Nat.succ" in hover_text

        term_goal = await mcp_client.call_tool(
            "lean_term_goal",
            {
                "file_path": str(target_file),
                "line": line,
                "column": column,
            },
        )
        type_text = result_text(term_goal)
        assert "→" in type_text
        assert any(fragment in type_text for fragment in ("ℕ", "Nat"))
