"""Diagnostics, file outline, and code-action tools."""

from __future__ import annotations

from typing import Annotated, Literal

from leanclient.aio.convert import range_from_utf16
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import (
    get_client,
    get_serial_scratch_pool,
    open_synced,
    require_client_for_file,
)
from lean_lsp_mcp.models import (
    CodeAction,
    CodeActionEdit,
    CodeActionsResult,
    DiagnosticsResult,
    FileOutline,
    InteractiveDiagnosticsResult,
)
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.tool_registry import tool
from lean_lsp_mcp.utils import get_declaration_range

_SEVERITY_LEVELS = ["error", "warning", "info", "hint"]


def _flatten_severity_schema(schema: dict) -> None:
    """Emit ``severity`` as a flat enum, satisfying every provider's validator.

    ``Optional[Literal[...]]`` generates an ``anyOf`` union with no top-level
    ``type``. Google Gemini/Vertex *requires* a top-level ``type`` (#185), while
    Moonshot/Kimi *forbids* a top-level ``type`` whenever ``anyOf`` is present
    (#213). A flat, nullable-by-omission enum (no ``anyOf``, explicit ``type``)
    is accepted by both, as well as by Anthropic and OpenAI.
    """
    schema.pop("anyOf", None)
    schema["type"] = "string"
    schema["enum"] = list(_SEVERITY_LEVELS)


async def _append_unique_code_actions(
    client,
    rel_path: str,
    ranges: list[tuple[int, int, int, int]],
    actions: list[dict],
    seen: set[str],
) -> None:
    for start_line, start_column, end_line, end_column in ranges:
        for action in await client.code_actions(
            rel_path, start_line, start_column, end_line, end_column
        ):
            title = action.get("title", "")
            if title not in seen:
                seen.add(title)
                actions.append(action)


async def _resolve_code_action(client, action: dict) -> dict | None:
    if "edit" in action:
        return action
    try:
        return await client.code_action_resolve(action)
    except Exception:
        return None


def _code_action_edits(
    resolved: dict, document_lines: list[str]
) -> list[CodeActionEdit]:
    edits: list[CodeActionEdit] = []
    for change in (resolved.get("edit") or {}).get("documentChanges", []):
        for edit in change.get("edits", []):
            edit_range = range_from_utf16(document_lines, edit["range"])
            edits.append(
                CodeActionEdit(
                    new_text=edit["newText"],
                    start_line=edit_range["start"]["line"] + 1,
                    start_column=edit_range["start"]["character"] + 1,
                    end_line=edit_range["end"]["line"] + 1,
                    end_column=edit_range["end"]["character"] + 1,
                )
            )
    return edits


@tool(
    "lean_file_outline",
    annotations=ToolAnnotations(
        title="File Outline",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def file_outline(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    max_declarations: Annotated[
        int | None, Field(description="Max declarations to return", ge=1)
    ] = None,
) -> FileOutline:
    """Get imports and declarations with type signatures. Token-efficient."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    pool = get_serial_scratch_pool(ctx)
    return await generate_outline_data(client, pool, rel_path, max_declarations)


@tool(
    "lean_diagnostic_messages",
    annotations=ToolAnnotations(
        title="Diagnostics",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def diagnostic_messages(
    ctx: server.ToolContext,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    start_line: Annotated[
        int | None, Field(description="Filter from line", ge=1)
    ] = None,
    end_line: Annotated[int | None, Field(description="Filter to line", ge=1)] = None,
    declaration_name: Annotated[
        str | None, Field(description="Filter to declaration (slow)")
    ] = None,
    interactive: Annotated[
        bool,
        Field(
            description="Returns verbose nested TaggedText with embedded widgets. Only use when plain text is insufficient. For 'Try This' suggestions, prefer lean_code_actions."
        ),
    ] = False,
    timeout_s: Annotated[
        float | None,
        Field(
            description=(
                "Max seconds to wait for elaboration. On timeout returns "
                "partial=true with still_elaborating_lines - poll again. "
                "Omit to wait for full elaboration."
            ),
            ge=1,
        ),
    ] = None,
    severity: Annotated[
        Literal["error", "warning", "info", "hint"] | None,
        # Flatten the emitted schema to a single enum with an explicit
        # top-level `type` and no `anyOf`. Gemini/Vertex requires the `type`
        # (#185); Moonshot/Kimi rejects `type` alongside `anyOf` (#213). See
        # _flatten_severity_schema.
        Field(
            description="Filter by severity level. Returns all levels when omitted.",
            json_schema_extra=_flatten_severity_schema,
        ),
    ] = None,
) -> DiagnosticsResult | InteractiveDiagnosticsResult:
    """Get compiler diagnostics (errors, warnings, infos) for a Lean file."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    await open_synced(ctx, rel_path)

    # If declaration_name is provided, get its range and use that for filtering
    if declaration_name:
        decl_range = await get_declaration_range(client, rel_path, declaration_name)
        if decl_range is None:
            raise server.LeanToolError(
                f"Declaration '{declaration_name}' not found in file."
            )
        start_line, end_line = decl_range

    # Convert 1-indexed to 0-indexed
    start_line_0 = (start_line - 1) if start_line is not None else None
    end_line_0 = (end_line - 1) if end_line is not None else None

    if interactive:
        line_range = None
        if start_line_0 is not None or end_line_0 is not None:
            line_range = {
                "start": start_line_0 or 0,
                "end": (end_line_0 + 1) if end_line_0 is not None else 2**30,
            }
        diagnostics = await client.rpc_call(
            rel_path,
            start_line_0 or 0,
            0,
            "Lean.Widget.getInteractiveDiagnostics",
            {"lineRange": line_range} if line_range else {},
        )
        return InteractiveDiagnosticsResult(diagnostics=diagnostics or [])

    report = await client.diagnostics(
        rel_path, fresh=True, timeout=timeout_s, partial_ok=True
    )
    processing_lines = [
        [r["start"]["line"] + 1, r["end"]["line"] + 1] for r in report.processing_ranges
    ] or None

    items = report.items
    if start_line_0 is not None or end_line_0 is not None:
        items = server._filter_diagnostics_by_line_range(
            items,
            start_line_0 if start_line_0 is not None else 0,
            end_line_0 if end_line_0 is not None else 2**30,
        )

    return server._process_diagnostics(
        items,
        build_success=not report.has_errors and not report.fatal_error,
        severity=severity,
        timed_out=report.partial,
        partial=report.partial,
        processing_lines=processing_lines,
    )


@tool(
    "lean_code_actions",
    annotations=ToolAnnotations(
        title="Code Actions",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def code_actions(
    ctx: server.ToolContext,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
) -> CodeActionsResult:
    """Get LSP code actions for a line. Returns resolved edits for TryThis suggestions (simp?, exact?, apply?) and other quick fixes."""
    rel_path = await require_client_for_file(ctx, file_path)

    client = get_client(ctx)
    await open_synced(ctx, rel_path)

    report = await client.diagnostics(rel_path)
    line_diags = server._filter_diagnostics_by_line_range(
        report.items, line - 1, line - 1
    )

    seen: set[str] = set()
    raw_actions: list[dict] = []
    diagnostic_ranges = []
    for diagnostic in line_diags:
        diagnostic_range = diagnostic.get("fullRange", diagnostic.get("range"))
        if diagnostic_range:
            start, end = diagnostic_range["start"], diagnostic_range["end"]
            diagnostic_ranges.append(
                (start["line"], start["character"], end["line"], end["character"])
            )
    await _append_unique_code_actions(
        client, rel_path, diagnostic_ranges, raw_actions, seen
    )

    # Fallback: if no diagnostics on the line, retry across the full line
    # range. Tactic `TryThis` suggestions (`simp?`, `exact?`, `apply?`) and
    # other `IdeView` quick-actions can be registered without an
    # accompanying diagnostic, so the diagnostic-driven scan misses them.
    if not raw_actions:
        lines = client.content(rel_path).splitlines()
        line_str = lines[line - 1] if 0 < line <= len(lines) else ""
        if line_str:
            await _append_unique_code_actions(
                client,
                rel_path,
                [(line - 1, 0, line - 1, len(line_str))],
                raw_actions,
                seen,
            )

    # Resolve and convert. Resolved edit ranges come straight from the LSP
    # (UTF-16 columns) — convert to codepoints against the document text.
    doc_lines = client.content(rel_path).splitlines()
    actions: list[CodeAction] = []
    for raw in raw_actions:
        resolved = await _resolve_code_action(client, raw)
        if resolved is None:
            continue
        actions.append(
            CodeAction(
                title=raw.get("title", ""),
                is_preferred=raw.get("isPreferred", False),
                edits=_code_action_edits(resolved, doc_lines),
            )
        )

    return CodeActionsResult(actions=actions)
