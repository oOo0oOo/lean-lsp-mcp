"""Project build tool."""

from __future__ import annotations

from typing import Annotated, Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import bind_lean_project_path
from lean_lsp_mcp.models import BuildResult


@server.mcp.tool(
    "lean_build",
    annotations=ToolAnnotations(
        title="Build Project",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def lsp_build(
    ctx: Context,
    lean_project_path: Annotated[
        Optional[str], Field(description="Path to Lean project")
    ] = None,
    clean: Annotated[bool, Field(description="Run lake clean first (slow)")] = False,
    fetch_cache: Annotated[
        bool, Field(description="Run lake exe cache get before building (slow)")
    ] = False,
    output_lines: Annotated[
        int, Field(description="Return last N lines of build log (0=none)")
    ] = 20,
) -> BuildResult:
    """Build the Lean project and restart LSP. Use only if needed (e.g. new imports)."""
    lifespan = ctx.request_context.lifespan_context
    configured_root = lifespan.lean_project_path

    if not lean_project_path:
        lean_project_path_obj = configured_root
    else:
        previous_root = configured_root
        try:
            lean_project_path_obj = bind_lean_project_path(ctx, lean_project_path)
        except ValueError as exc:
            raise server.LeanToolError(str(exc)) from exc
        if previous_root is not None and previous_root != lean_project_path_obj:
            await server._close_repl_for_project_switch(lifespan)

    if lean_project_path_obj is None:
        raise server.LeanToolError(
            "Lean project path not known yet. Provide `lean_project_path` explicitly or call another tool first."
        )

    async def build_factory() -> BuildResult:
        return await server._run_build(
            ctx, lean_project_path_obj, clean, fetch_cache, output_lines
        )

    app_ctx = ctx.request_context.lifespan_context
    coordinator = app_ctx.build_coordinator
    if coordinator is None or coordinator.mode == "allow":
        return await build_factory()
    return await coordinator.run(build_factory)
