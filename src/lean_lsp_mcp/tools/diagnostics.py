"""Diagnostics, file outline, and code-action tools."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
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
def file_outline(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    max_declarations: Annotated[
        Optional[int], Field(description="Max declarations to return", ge=1)
    ] = None,
) -> FileOutline:
    """Get imports and declarations with type signatures. Token-efficient."""
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    return generate_outline_data(client, rel_path, max_declarations)


@server.mcp.tool(
    "lean_diagnostic_messages",
    annotations=ToolAnnotations(
        title="Diagnostics",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def diagnostic_messages(
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
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    # If declaration_name is provided, get its range and use that for filtering
    if declaration_name:
        decl_range = get_declaration_range(client, rel_path, declaration_name)
        if decl_range is None:
            raise server.LeanToolError(
                f"Declaration '{declaration_name}' not found in file."
            )
        start_line, end_line = decl_range

    # Convert 1-indexed to 0-indexed for leanclient
    start_line_0 = (start_line - 1) if start_line is not None else None
    end_line_0 = (end_line - 1) if end_line is not None else None

    if interactive:
        diagnostics = client.get_interactive_diagnostics(
            rel_path, start_line=start_line_0, end_line=end_line_0
        )
        return InteractiveDiagnosticsResult(diagnostics=diagnostics)

    result = client.get_diagnostics(
        rel_path,
        start_line=start_line_0,
        end_line=end_line_0,
        inactivity_timeout=15.0,
    )

    return server._process_diagnostics(
        result.diagnostics,
        result.success,
        severity=severity,
        timed_out=getattr(result, "timed_out", False),
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
def code_actions(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
) -> CodeActionsResult:
    """Get LSP code actions for a line. Returns resolved edits for TryThis suggestions (simp?, exact?, apply?) and other quick fixes."""
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    # Get diagnostics on line to discover code action ranges
    diags = client.get_diagnostics(
        rel_path, start_line=line - 1, end_line=line - 1, inactivity_timeout=15.0
    )

    # Query code actions for each diagnostic's range, dedup by title
    seen: set[str] = set()
    raw_actions: list[dict] = []
    for diag in diags.diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if not r:
            continue
        s, e = r["start"], r["end"]
        for action in client.get_code_actions(
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
        line_str = ""
        try:
            resolved_path = server.resolve_file_path(ctx, file_path)
            line_text = resolved_path.read_text(encoding="utf-8").splitlines()
            line_str = line_text[line - 1] if 0 < line <= len(line_text) else ""
        except (
            OSError,
            IndexError,
            UnicodeDecodeError,
            ValueError,
            RuntimeError,
        ) as exc:
            # Path-resolution failures are common for files in
            # `.lake/packages/...` (dep paths that `setup_client_for_file`
            # accepts but `resolve_file_path(require_exists=True)` rejects).
            # `RuntimeError` covers CPython's symlink-loop raise from
            # ``Path.resolve(strict=True)``.
            # Log so production failures aren't silently invisible.
            server.logger.debug(
                "lean_code_actions fallback: could not read line text from %s: %s",
                file_path,
                exc,
            )
            line_str = ""
        if line_str:
            # LSP positions are UTF-16 code units, not Python codepoints —
            # surrogate-pair characters (e.g. `𝕜`) count as 2 units. Use the
            # UTF-16 length so the end-column reaches the actual end of the
            # line on those rare inputs.
            end_col = len(line_str.encode("utf-16-le")) // 2
            for action in client.get_code_actions(
                rel_path, line - 1, 0, line - 1, end_col
            ):
                if action.get("title", "") not in seen:
                    seen.add(action.get("title", ""))
                    raw_actions.append(action)

    # Resolve and convert
    actions: list[CodeAction] = []
    for raw in raw_actions:
        resolved = raw if "edit" in raw else client.get_code_action_resolve(raw)
        if isinstance(resolved, dict) and "error" in resolved:
            continue
        actions.append(
            CodeAction(
                title=raw.get("title", ""),
                is_preferred=raw.get("isPreferred", False),
                edits=[
                    CodeActionEdit(
                        new_text=edit["newText"],
                        start_line=edit["range"]["start"]["line"] + 1,
                        start_column=edit["range"]["start"]["character"] + 1,
                        end_line=edit["range"]["end"]["line"] + 1,
                        end_column=edit["range"]["end"]["character"] + 1,
                    )
                    for dc in resolved.get("edit", {}).get("documentChanges", [])
                    for edit in dc.get("edits", [])
                ],
            )
        )

    return CodeActionsResult(actions=actions)
