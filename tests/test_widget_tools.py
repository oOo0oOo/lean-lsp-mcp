from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


@pytest.fixture()
def widget_file(test_project_path: Path) -> Path:
    path = test_project_path / "WidgetSample.lean"
    if not path.exists():
        pytest.skip("Widget sample file missing")
    return path


@pytest.mark.asyncio
async def test_widget_tools(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    content = widget_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    widget_line_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("#widget")
    )
    column = lines[widget_line_index].index("#widget") + 1

    async with mcp_client_factory() as client:
        widgets_result = await client.call_tool(
            "lean_widgets",
            {
                "file_path": str(widget_file),
                "line": widget_line_index + 1,
                "column": column,
            },
        )
        structured = widgets_result.structuredContent
        assert structured is not None, "Expected structured content from lean_widgets"
        items = structured.get("items")
        assert isinstance(items, list), "Expected widget instances list"
        assert items, "Expected at least one widget instance"

        widget = items[0]
        widget_source = await client.call_tool(
            "lean_widget_source",
            {
                "file_path": str(widget_file),
                "line": widget_line_index + 1,
                "column": column,
                "widget": widget,
            },
        )
        source_structured = widget_source.structuredContent
        assert source_structured is not None, "Expected structured content from source"
        assert "sourcetext" in source_structured
        assert "export default function Hello" in source_structured["sourcetext"]

        diagnostics_result = await client.call_tool(
            "lean_interactive_diagnostics",
            {
                "file_path": str(widget_file),
                "start_line": widget_line_index + 1,
                "end_line": widget_line_index + 1,
            },
        )
        diag_structured = diagnostics_result.structuredContent
        assert diag_structured is not None, "Expected structured content from diagnostics"
        assert isinstance(diag_structured.get("diagnostics"), list)
        assert isinstance(diag_structured.get("widgets"), list)
