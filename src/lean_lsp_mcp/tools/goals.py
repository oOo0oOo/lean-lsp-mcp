"""Proof-goal inspection tools."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.models import GoalState, TermGoalState


@server.mcp.tool(
    "lean_goal",
    annotations=ToolAnnotations(
        title="Proof Goals",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def goal(
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
    showing how the tactic transforms the state. "no goals" = proof complete.
    """
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise server.LeanToolError(
            f"Line {line} out of range (file has {len(lines)} lines)"
        )

    line_context = lines[line - 1]

    if column is None:
        column_end = len(line_context)
        column_start = next(
            (i for i, c in enumerate(line_context) if not c.isspace()), 0
        )
        goal_start = server._get_goal_response(client, rel_path, line - 1, column_start)
        server.check_lsp_response(goal_start, "get_goal", allow_none=True)
        goal_end = server._get_goal_response(client, rel_path, line - 1, column_end)
        goals_before = server.extract_goals_list(goal_start)
        goals_after = server.extract_goals_list(goal_end)

        if format == "structured":
            goals_before = [server._goal_to_structured(g) for g in goals_before]
            goals_after = [server._goal_to_structured(g) for g in goals_after]

        return GoalState(
            line_context=line_context,
            goals_before=goals_before,
            goals_after=goals_after,
        )
    else:
        goal_result = server._get_goal_response(client, rel_path, line - 1, column - 1)
        server.check_lsp_response(goal_result, "get_goal", allow_none=True)
        goals = server.extract_goals_list(goal_result)
        if format == "structured":
            goals = [server._goal_to_structured(g) for g in goals]
        return GoalState(
            line_context=line_context,
            goals=goals,
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
def term_goal(
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
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise server.LeanToolError(
            f"Line {line} out of range (file has {len(lines)} lines)"
        )

    line_context = lines[line - 1]
    if column is None:
        column = max(len(line_context), 1)

    term_goal_result = client.get_term_goal(rel_path, line - 1, column - 1)
    server.check_lsp_response(term_goal_result, "get_term_goal", allow_none=True)
    expected_type = None
    if term_goal_result is not None:
        rendered = term_goal_result.get("goal")
        if rendered:
            expected_type = rendered.replace("```lean\n", "").replace("\n```", "")

    return TermGoalState(line_context=line_context, expected_type=expected_type)
