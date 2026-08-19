"""Navigation and introspection tools: hover, completions, declaration, references."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import (
    get_client,
    get_path_policy,
    open_synced,
    require_client_for_file,
)
from lean_lsp_mcp.models import (
    CompletionItem,
    CompletionsResult,
    DeclarationInfo,
    HoverInfo,
    ReferenceLocation,
    ReferencesResult,
)
from lean_lsp_mcp.tool_registry import tool
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
)


def _completion_prefix(content: str, line: int, column: int) -> str:
    lines = content.splitlines()
    if line < 1 or line > len(lines):
        return ""
    text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
    if text_before_cursor.endswith("."):
        return ""
    return re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[-1].lower()


def _sort_completions(
    completions: list[dict], content: str, line: int, column: int
) -> None:
    if any("sortText" in completion for completion in completions):
        completions.sort(key=lambda completion: completion.get("sortText", "\uffff"))
        return

    prefix = _completion_prefix(content, line, column)
    if not prefix:
        completions.sort(key=lambda completion: completion["label"].lower())
        return

    def key(completion: dict) -> tuple[int, str]:
        label = completion["label"].lower()
        if label.startswith(prefix):
            return 0, label
        if prefix in label:
            return 1, label
        return 2, label

    completions.sort(key=key)


async def _resolve_completion_details(
    client, completions: list[dict], count: int
) -> list[dict]:
    async def resolve(completion: dict) -> dict:
        if completion.get("detail"):
            return completion
        try:
            return await client.completion_resolve(completion)
        except Exception:
            return completion

    if count <= 0:
        return completions
    resolved = await asyncio.gather(*(resolve(item) for item in completions[:count]))
    return [*resolved, *completions[count:]]


def _completion_items(completions: list[dict]) -> list[CompletionItem]:
    return [
        CompletionItem(
            label=completion["label"],
            kind=(
                COMPLETION_KIND.get(kind) if (kind := completion.get("kind")) else None
            ),
            detail=completion.get("detail"),
        )
        for completion in completions
    ]


@tool(
    "lean_hover_info",
    annotations=ToolAnnotations(
        title="Hover Info",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def hover(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int,
        Field(description="Column at START of identifier (1-indexed characters)", ge=1),
    ],
) -> HoverInfo:
    """Get type signature and docs for a symbol. Essential for understanding APIs."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    await open_synced(ctx, rel_path)
    file_content = client.content(rel_path)
    hover_info = await client.hover(rel_path, line - 1, column - 1)
    if hover_info is None:
        raise server.LeanToolError(
            f"No hover information at line {line}, column {column}"
        )

    # Get the symbol and the hover information
    h_range = hover_info.get("range")
    symbol = extract_range(file_content, h_range) or ""
    info = hover_info["contents"].get("value", "No hover information available.")
    info = info.replace("```lean\n", "").replace("\n```", "").strip()

    # Add diagnostics if available
    report = await client.diagnostics(rel_path, fresh=False)
    filtered = filter_diagnostics_by_position(report.items, line - 1, column - 1)

    return HoverInfo(
        symbol=symbol,
        info=info,
        diagnostics=server._to_diagnostic_messages(filtered),
    )


@tool(
    "lean_completions",
    annotations=ToolAnnotations(
        title="Completions",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def completions(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int, Field(description="Column number (1-indexed characters)", ge=1)
    ],
    max_completions: Annotated[int, Field(description="Max completions", ge=1)] = 32,
    resolve_details: Annotated[
        int,
        Field(description="Fetch type signatures for the top N results", ge=0),
    ] = 8,
) -> CompletionsResult:
    """Get IDE autocompletions. Use on INCOMPLETE code (after `.` or partial name)."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    await open_synced(ctx, rel_path)
    content = client.content(rel_path)
    raw_completions = await client.completions(rel_path, line - 1, column - 1)
    raw_completions = [c for c in raw_completions if "label" in c]

    if not raw_completions:
        return CompletionsResult(items=[])

    _sort_completions(raw_completions, content, line, column)
    raw_completions = raw_completions[:max_completions]
    raw_completions = await _resolve_completion_details(
        client, raw_completions, resolve_details
    )
    return CompletionsResult(items=_completion_items(raw_completions))


@tool(
    "lean_declaration_file",
    annotations=ToolAnnotations(
        title="Declaration Source",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def declaration_file(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    symbol: Annotated[
        str, Field(description="Symbol (case sensitive, must be in file)")
    ],
    context_lines: Annotated[
        int,
        Field(description="Lines of context around the declaration", ge=0),
    ] = 20,
    full_file: Annotated[
        bool, Field(description="Return the entire declaration file (large!)")
    ] = False,
) -> DeclarationInfo:
    """Get the source of a symbol's declaration (declaration slice + context).

    Set full_file=True for the whole file (can be very large)."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    await open_synced(ctx, rel_path)
    orig_file_content = client.content(rel_path)

    # Find the first occurence of the symbol (line and column) in the file
    position = find_start_position(orig_file_content, symbol)
    if not position:
        raise server.LeanToolError(
            f"Symbol `{symbol}` (case sensitive) not found in file. Add it first."
        )

    locations = await client.goto(
        "declaration", rel_path, position["line"], position["column"]
    )
    if not locations:
        raise server.LeanToolError(f"No declaration available for `{symbol}`.")

    decl = locations[0]
    local_path = decl.get("path", "")

    try:
        policy = get_path_policy(ctx)
        abs_path = policy.validate_path(
            Path(local_path)
            if Path(local_path).is_absolute()
            else Path(client.project_path) / local_path
        )
    except ValueError as exc:
        raise server.LeanToolError(str(exc)) from exc

    if not abs_path.exists():
        raise server.LeanToolError(
            f"Could not open declaration file `{abs_path}` for `{symbol}`."
        )

    file_content = server.get_file_contents(abs_path)
    file_lines = file_content.splitlines()
    total_lines = len(file_lines)

    if full_file:
        return DeclarationInfo(
            file_path=policy.display_path(abs_path),
            content=file_content,
            start_line=1,
            end_line=total_lines,
            total_lines=total_lines,
        )

    target_range = decl.get("targetRange") or decl.get("range") or {}
    decl_start = target_range.get("start", {}).get("line", 0)
    decl_end = target_range.get("end", {}).get("line", decl_start)
    start = max(0, decl_start - context_lines)
    end = min(total_lines, decl_end + 1 + context_lines)

    return DeclarationInfo(
        file_path=policy.display_path(abs_path),
        content="\n".join(file_lines[start:end]),
        start_line=start + 1,
        end_line=end,
        total_lines=total_lines,
    )


@tool(
    "lean_references",
    annotations=ToolAnnotations(
        title="Find References",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def references(
    ctx: server.ToolContext,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int, Field(description="Column at START of identifier (1-indexed)", ge=1)
    ],
    max_results: Annotated[
        int | None,
        Field(description="Max locations to return (default 50)", ge=1),
    ] = 50,
) -> ReferencesResult:
    """Find all references to a symbol (including the declaration). Position cursor at the symbol."""
    rel_path = await require_client_for_file(
        ctx,
        file_path,
        error_message="Invalid Lean file path: Unable to start LSP server or load file",
    )

    client = get_client(ctx)
    await open_synced(ctx, rel_path)

    try:
        raw_refs = await client.references(
            rel_path, line - 1, column - 1, include_declaration=True
        )
    except Exception as e:
        raise server.LeanToolError(f"Failed to get references: {e}")

    total = len(raw_refs)
    if max_results is not None:
        raw_refs = raw_refs[:max_results]

    items: list[ReferenceLocation] = []
    try:
        policy = get_path_policy(ctx)
    except ValueError as exc:
        raise server.LeanToolError(str(exc)) from exc
    for ref in raw_refs:
        r = ref.get("range", {})
        display = ref.get("path", "")
        if display:
            candidate = (
                Path(display)
                if Path(display).is_absolute()
                else Path(client.project_path) / display
            )
            try:
                display = policy.display_path(candidate)
            except ValueError:
                continue
        items.append(
            ReferenceLocation(
                file_path=display,
                line=r.get("start", {}).get("line", 0) + 1,
                column=r.get("start", {}).get("character", 0) + 1,
                end_line=r.get("end", {}).get("line", 0) + 1,
                end_column=r.get("end", {}).get("character", 0) + 1,
            )
        )

    return ReferencesResult(items=items, total=total)
