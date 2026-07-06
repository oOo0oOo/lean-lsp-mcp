"""Widget / infoview tools."""

from __future__ import annotations

from typing import Annotated

from leanclient.aio import AsyncLeanLSPClient
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import open_synced
from lean_lsp_mcp.models import WidgetsResult, WidgetSourceResult


@server.mcp.tool(
    "lean_get_widgets",
    annotations=ToolAnnotations(
        title="Get Widgets",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_widgets(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
) -> WidgetsResult:
    """Get panel widgets at a position (proof visualizations, #html, custom widgets). Returns raw widget data - may be large."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    await open_synced(ctx, rel_path)
    widgets = await client.get_widgets(rel_path, line - 1, column - 1)
    return WidgetsResult(widgets=widgets)


@server.mcp.tool(
    "lean_get_widget_source",
    annotations=ToolAnnotations(
        title="Widget Source",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_widget_source(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    javascript_hash: Annotated[
        str, Field(description="javascriptHash from a widget instance")
    ],
) -> WidgetSourceResult:
    """Get JavaScript source of a widget by hash. Useful for understanding custom widget rendering logic. Returns full JS module - may be large."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    await open_synced(ctx, rel_path, wait=True)
    source = await client.get_widget_source(rel_path, 0, 0, javascript_hash)
    if source is None:
        raise server.LeanToolError(
            f"Widget source not found for hash `{javascript_hash}`."
        )
    return WidgetSourceResult(source=source)
