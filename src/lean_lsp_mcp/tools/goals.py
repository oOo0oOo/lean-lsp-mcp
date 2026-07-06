"""Proof-goal inspection tools."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from leanclient.aio import AsyncLeanLSPClient
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import open_synced
from lean_lsp_mcp.models import GoalState, StructuredGoal, TermGoalState


@server.mcp.tool(
    "lean_goal",
    annotations=ToolAnnotations(
        title="Proof Goals",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def goal(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int],
        Field(description="Column (1-indexed). Omit for before/after", ge=1),
    ] = None,
    format: Annotated[
        Literal["text", "structured"],
        Field(description="Output format: 'text' (default) or 'structured'"),
    ] = "text",
) -> GoalState:
    """Get proof goals at a position. MOST IMPORTANT tool - use often!

    Omit column to see goals_before (line start) and goals_after (line end),
    showing how the tactic transforms the state. status='complete' means the
    proof is finished at this point; status='no_goal_at_position' means the
    position carries no proof state (e.g. outside a proof).
    """
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    await open_synced(ctx, rel_path)
    content = client.content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise server.LeanToolError(
            f"Line {line} out of range (file has {len(lines)} lines)"
        )

    line_context = lines[line - 1]
    structured = format == "structured"

    def render(goals: list[str]) -> list[str | StructuredGoal]:
        return [server._goal_to_structured(g) if structured else g for g in goals]

    if column is None:
        column_start = next(
            (i for i, c in enumerate(line_context) if not c.isspace()), 0
        )
        column_end = len(line_context)
        # First call barriers (fresh); second reuses the elaborated state.
        goal_start = await client.goal(rel_path, line - 1, column_start)
        goal_end = await client.goal(rel_path, line - 1, column_end, fresh=False)

        return GoalState(
            line_context=line_context,
            goals_before=render(server._goal_strings(goal_start)),
            goals_after=render(server._goal_strings(goal_end)),
        )

    result = await client.goal(rel_path, line - 1, column - 1)
    status = "no_goal_at_position" if result.status == "no_goal" else result.status
    return GoalState(
        line_context=line_context,
        goals=render(result.goals) if result.status == "goals" else [],
        status=status,
    )


@server.mcp.tool(
    "lean_term_goal",
    annotations=ToolAnnotations(
        title="Term Goal",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def term_goal(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int], Field(description="Column (defaults to end of line)", ge=1)
    ] = None,
) -> TermGoalState:
    """Get the expected type at a position."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    await open_synced(ctx, rel_path)
    content = client.content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise server.LeanToolError(
            f"Line {line} out of range (file has {len(lines)} lines)"
        )

    line_context = lines[line - 1]
    if column is None:
        column = max(len(line_context), 1)

    term_goal_result = await client.term_goal(rel_path, line - 1, column - 1)
    expected_type = None
    if term_goal_result is not None:
        rendered = term_goal_result.get("goal")
        if rendered:
            expected_type = rendered.replace("```lean\n", "").replace("\n```", "")

    return TermGoalState(line_context=line_context, expected_type=expected_type)
