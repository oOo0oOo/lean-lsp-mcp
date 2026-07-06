"""Diagnostics, file outline, and code-action tools."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from leanclient.aio import (
    AsyncLeanLSPClient,
    LeanRequestTimeout,
)
from leanclient.aio.convert import range_from_utf16
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import get_scratch_pool, open_synced
from lean_lsp_mcp.models import (
    CodeAction,
    CodeActionEdit,
    CodeActionsResult,
    DiagnosticsResult,
    FileOutline,
    InteractiveDiagnosticsResult,
)
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.utils import get_declaration_range


@server.mcp.tool(
    "lean_file_outline",
    annotations=ToolAnnotations(
        title="File Outline",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def file_outline(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    max_declarations: Annotated[
        Optional[int], Field(description="Max declarations to return", ge=1)
    ] = None,
) -> FileOutline:
    """Get imports and declarations with type signatures. Token-efficient."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    pool = get_scratch_pool(ctx)
    return await generate_outline_data(client, pool, rel_path, max_declarations)


@server.mcp.tool(
    "lean_diagnostic_messages",
    annotations=ToolAnnotations(
        title="Diagnostics",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def diagnostic_messages(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    start_line: Annotated[
        Optional[int], Field(description="Filter from line", ge=1)
    ] = None,
    end_line: Annotated[
        Optional[int], Field(description="Filter to line", ge=1)
    ] = None,
    declaration_name: Annotated[
        Optional[str], Field(description="Filter to declaration (slow)")
    ] = None,
    interactive: Annotated[
        bool,
        Field(
            description="Returns verbose nested TaggedText with embedded widgets. Only use when plain text is insufficient. For 'Try This' suggestions, prefer lean_code_actions."
        ),
    ] = False,
    severity: Annotated[
        Optional[Literal["error", "warning", "info", "hint"]],
        # json_schema_extra forces an explicit top-level `type` onto the
        # property schema. Without it the emitted schema is a bare
        # anyOf/enum union with no `type`, which Google Gemini/Vertex
        # function-calling rejects ("schema didn't specify the schema type
        # field"). See issue #185.
        Field(
            description="Filter by severity level. Returns all levels when omitted.",
            json_schema_extra={"type": "string"},
        ),
    ] = None,
) -> DiagnosticsResult | InteractiveDiagnosticsResult:
    """Get compiler diagnostics (errors, warnings, infos) for a Lean file."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
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

    timed_out = False
    try:
        report = await client.diagnostics(rel_path)
    except LeanRequestTimeout:
        timed_out = True
        report = await client.diagnostics(rel_path, fresh=False)

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
        timed_out=timed_out,
    )


@server.mcp.tool(
    "lean_code_actions",
    annotations=ToolAnnotations(
        title="Code Actions",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def code_actions(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
) -> CodeActionsResult:
    """Get LSP code actions for a line. Returns resolved edits for TryThis suggestions (simp?, exact?, apply?) and other quick fixes."""
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    await open_synced(ctx, rel_path)

    report = await client.diagnostics(rel_path)
    line_diags = server._filter_diagnostics_by_line_range(
        report.items, line - 1, line - 1
    )

    # Query code actions for each diagnostic's range, dedup by title.
    # Ranges from the aio client are codepoint columns, matching the
    # codepoint-based code_actions() API.
    seen: set[str] = set()
    raw_actions: list[dict] = []
    for diag in line_diags:
        r = diag.get("fullRange", diag.get("range"))
        if not r:
            continue
        s, e = r["start"], r["end"]
        for action in await client.code_actions(
            rel_path, s["line"], s["character"], e["line"], e["character"]
        ):
            if action.get("title", "") not in seen:
                seen.add(action.get("title", ""))
                raw_actions.append(action)

    # Fallback: if no diagnostics on the line, retry across the full line
    # range. Tactic `TryThis` suggestions (`simp?`, `exact?`, `apply?`) and
    # other `IdeView` quick-actions can be registered without an
    # accompanying diagnostic, so the diagnostic-driven scan misses them.
    if not raw_actions:
        lines = client.content(rel_path).splitlines()
        line_str = lines[line - 1] if 0 < line <= len(lines) else ""
        if line_str:
            for action in await client.code_actions(
                rel_path, line - 1, 0, line - 1, len(line_str)
            ):
                if action.get("title", "") not in seen:
                    seen.add(action.get("title", ""))
                    raw_actions.append(action)

    # Resolve and convert. Resolved edit ranges come straight from the LSP
    # (UTF-16 columns) — convert to codepoints against the document text.
    doc_lines = client.content(rel_path).splitlines()
    actions: list[CodeAction] = []
    for raw in raw_actions:
        if "edit" in raw:
            resolved = raw
        else:
            try:
                resolved = await client.code_action_resolve(raw)
            except Exception:
                continue
        edits = []
        for dc in (resolved.get("edit") or {}).get("documentChanges", []):
            for edit in dc.get("edits", []):
                rng = range_from_utf16(doc_lines, edit["range"])
                edits.append(
                    CodeActionEdit(
                        new_text=edit["newText"],
                        start_line=rng["start"]["line"] + 1,
                        start_column=rng["start"]["character"] + 1,
                        end_line=rng["end"]["line"] + 1,
                        end_column=rng["end"]["character"] + 1,
                    )
                )
        actions.append(
            CodeAction(
                title=raw.get("title", ""),
                is_preferred=raw.get("isPreferred", False),
                edits=edits,
            )
        )

    return CodeActionsResult(actions=actions)
