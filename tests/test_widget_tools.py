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

            open ProofWidgets in
            #html <b>Hello from widget test</b>
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


@pytest.mark.asyncio
async def test_get_interactive_diagnostics(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_get_interactive_diagnostics",
            {
                "file_path": str(widget_file),
            },
        )
        data = result_json(result)
        assert "diagnostics" in data
        assert isinstance(data["diagnostics"], list)


@pytest.mark.asyncio
async def test_get_interactive_diagnostics_with_line_range(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    widget_file: Path,
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_get_interactive_diagnostics",
            {
                "file_path": str(widget_file),
                "start_line": 1,
                "end_line": 4,
            },
        )
        data = result_json(result)
        assert "diagnostics" in data
        assert isinstance(data["diagnostics"], list)
