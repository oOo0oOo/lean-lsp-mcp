from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.fixture()
def widget_file(test_project_path: Path) -> Path:
    path = test_project_path / "WidgetTest.lean"
    path.write_text(
        textwrap.dedent("""\
            import ProofWidgets.Component.HtmlDisplay
            open ProofWidgets

            #html Html.element "b" #[] #[.text "Hello widget"]
        """),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_get_widgets(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_get_widgets",
            {
                "file_path": str(widget_file),
                "line": 4,
                "column": 1,
            },
        )
        data = result_json(result)
        assert "widgets" in data
        assert len(data["widgets"]) > 0
        widget = data["widgets"][0]
        assert widget["id"] == "ProofWidgets.HtmlDisplayPanel"
        assert "props" in widget


@pytest.mark.asyncio
async def test_diagnostic_messages_interactive(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(widget_file),
                "interactive": True,
            },
        )
        data = result_json(result)
        assert "diagnostics" in data
        assert isinstance(data["diagnostics"], list)


@pytest.mark.asyncio
async def test_diagnostic_messages_interactive_with_line_range(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_diagnostic_messages",
            {
                "file_path": str(widget_file),
                "start_line": 1,
                "end_line": 4,
                "interactive": True,
            },
        )
        data = result_json(result)
        assert "diagnostics" in data
        assert isinstance(data["diagnostics"], list)


@pytest.mark.asyncio
async def test_get_widget_source(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        # First get a widget to obtain its hash
        widgets_result = await client.call_tool(
            "lean_get_widgets",
            {"file_path": str(widget_file), "line": 4, "column": 1},
        )
        widgets = result_json(widgets_result)["widgets"]
        assert len(widgets) > 0
        js_hash = widgets[0]["javascriptHash"]

        result = await client.call_tool(
            "lean_get_widget_source",
            {
                "file_path": str(widget_file),
                "javascript_hash": js_hash,
            },
        )
        data = result_json(result)
        assert "source" in data
