"""Proof-goal inspection tools."""

from __future__ import annotations

from typing import Annotated, Literal

from leanclient.aio import LeanRequestTimeout
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import get_client, open_synced, require_client_for_file
from lean_lsp_mcp.models import GoalState, StructuredGoal, TermGoalState
from lean_lsp_mcp.tool_registry import tool


@tool(
    "lean_goal",
    annotations=ToolAnnotations(
        title="Proof Goals",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def goal(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int | None,
        Field(description="Column (1-indexed). Omit for before/after", ge=1),
    ] = None,
    format: Annotated[
        Literal["text", "structured"],
        Field(description="Output format: 'text' (default) or 'structured'"),
    ] = "text",
    timeout_s: Annotated[
        float | None,
        Field(
            description=(
                "Max seconds to wait for elaboration. On timeout returns "
                "status='still_elaborating' - poll again."
            ),
            ge=1,
        ),
    ] = None,
) -> GoalState:
    """Get proof goals at a position. MOST IMPORTANT tool - use often!

    Omit column to see goals_before (line start) and goals_after (line end),
    showing how the tactic transforms the state. status='complete' means the
    proof is finished at this point; status='no_goal_at_position' means the
    position carries no proof state (e.g. outside a proof).
    """
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
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

    try:
        await client.barrier(rel_path, timeout=timeout_s)
    except LeanRequestTimeout:
        return GoalState(
            line_context=line_context,
            goals=[],
            status="still_elaborating",
        )

    if column is None:
        column_start = next(
            (i for i, c in enumerate(line_context) if not c.isspace()), 0
        )
        column_end = len(line_context)
        # Barrier already passed above; both queries reuse the elaborated state.
        goal_start = await client.goal(rel_path, line - 1, column_start, fresh=False)
        goal_end = await client.goal(rel_path, line - 1, column_end, fresh=False)

        return GoalState(
            line_context=line_context,
            goals_before=render(server._goal_strings(goal_start)),
            goals_after=render(server._goal_strings(goal_end)),
        )

    result = await client.goal(rel_path, line - 1, column - 1, fresh=False)
    status = "no_goal_at_position" if result.status == "no_goal" else result.status
    return GoalState(
        line_context=line_context,
        goals=render(result.goals) if result.status == "goals" else [],
        status=status,
    )


@tool(
    "lean_term_goal",
    annotations=ToolAnnotations(
        title="Term Goal",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def term_goal(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int | None, Field(description="Column (defaults to end of line)", ge=1)
    ] = None,
) -> TermGoalState:
    """Get the expected type at a position."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
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
