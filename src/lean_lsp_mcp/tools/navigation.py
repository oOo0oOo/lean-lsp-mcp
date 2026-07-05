"""Navigation and introspection tools: hover, completions, declaration, references."""

from __future__ import annotations

import re
from typing import Annotated, List

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.models import (
    CompletionItem,
    CompletionsResult,
    DeclarationInfo,
    HoverInfo,
    ReferenceLocation,
    ReferencesResult,
)
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    check_lsp_response,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
)


@server.mcp.tool(
    "lean_hover_info",
    annotations=ToolAnnotations(
        title="Hover Info",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def hover(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column at START of identifier", ge=1)],
) -> HoverInfo:
    """Get type signature and docs for a symbol. Essential for understanding APIs."""
    try:
        with server.lsp_client_for_file(ctx, file_path) as lsp:
            client = lsp.client
            rel_path = lsp.rel_path
            file_content = client.get_file_content(rel_path)
            hover_info = client.get_hover(rel_path, line - 1, column - 1)
            check_lsp_response(hover_info, "get_hover", allow_none=True)
            if hover_info is None:
                raise server.LeanToolError(
                    f"No hover information at line {line}, column {column}"
                )

            # Get the symbol and the hover information
            h_range = hover_info.get("range")
            symbol = extract_range(file_content, h_range) or ""
            info = hover_info["contents"].get(
                "value", "No hover information available."
            )
            info = info.replace("```lean\n", "").replace("\n```", "").strip()

            diagnostics = client.get_diagnostics(rel_path)
            check_lsp_response(diagnostics, "get_diagnostics")
            filtered = filter_diagnostics_by_position(diagnostics, line - 1, column - 1)

            return HoverInfo(
                symbol=symbol,
                info=info,
                diagnostics=server._to_diagnostic_messages(filtered),
            )
    except server.InvalidLeanFilePathError:
        server._raise_invalid_path(file_path)


@server.mcp.tool(
    "lean_completions",
    annotations=ToolAnnotations(
        title="Completions",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def completions(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    max_completions: Annotated[int, Field(description="Max completions", ge=1)] = 32,
) -> CompletionsResult:
    """Get IDE autocompletions. Use on INCOMPLETE code (after `.` or partial name)."""
    try:
        with server.lsp_client_for_file(ctx, file_path) as lsp:
            client = lsp.client
            rel_path = lsp.rel_path
            content = client.get_file_content(rel_path)
            raw_completions = client.get_completions(rel_path, line - 1, column - 1)
            check_lsp_response(raw_completions, "get_completions")

            # Convert to CompletionItem models
            items: List[CompletionItem] = []
            for c in raw_completions:
                if "label" not in c:
                    continue
                kind_int = c.get("kind")
                kind_str = COMPLETION_KIND.get(kind_int) if kind_int else None
                items.append(
                    CompletionItem(
                        label=c["label"],
                        kind=kind_str,
                        detail=c.get("detail"),
                    )
                )

            if not items:
                return CompletionsResult(items=[])

            # Find the sort term: The last word/identifier before the cursor
            lines = content.splitlines()
            prefix = ""
            if 0 < line <= len(lines):
                text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
                if not text_before_cursor.endswith("."):
                    prefix = re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[
                        -1
                    ].lower()

            # Sort completions: prefix matches first, then contains, then alphabetical
            if prefix:

                def sort_key(item: CompletionItem):
                    label_lower = item.label.lower()
                    if label_lower.startswith(prefix):
                        return (0, label_lower)
                    elif prefix in label_lower:
                        return (1, label_lower)
                    else:
                        return (2, label_lower)

                items.sort(key=sort_key)
            else:
                items.sort(key=lambda x: x.label.lower())

            # Truncate if too many results
            return CompletionsResult(items=items[:max_completions])
    except server.InvalidLeanFilePathError:
        server._raise_invalid_path(file_path)


@server.mcp.tool(
    "lean_declaration_file",
    annotations=ToolAnnotations(
        title="Declaration Source",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def declaration_file(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    symbol: Annotated[
        str, Field(description="Symbol (case sensitive, must be in file)")
    ],
) -> DeclarationInfo:
    """Get file where a symbol is declared. Symbol must be present in file first."""
    try:
        with server.lsp_client_for_file(ctx, file_path) as lsp:
            client = lsp.client
            rel_path = lsp.rel_path
            orig_file_content = client.get_file_content(rel_path)

            # Find the first occurence of the symbol (line and column) in the file
            position = find_start_position(orig_file_content, symbol)
            if not position:
                raise server.LeanToolError(
                    f"Symbol `{symbol}` (case sensitive) not found in file. Add it first."
                )

            declaration = client.get_declarations(
                rel_path, position["line"], position["column"]
            )

            if len(declaration) == 0:
                raise server.LeanToolError(f"No declaration available for `{symbol}`.")

            # Load the declaration file
            decl = declaration[0]
            uri = decl.get("targetUri") or decl.get("uri")

            policy = lsp.path_policy
            try:
                abs_path = policy.validate_path(client._uri_to_abs(uri))
            except ValueError as exc:
                raise server.LeanToolError(str(exc)) from exc
    except server.InvalidLeanFilePathError:
        server._raise_invalid_path(file_path)

    if not abs_path.exists():
        raise server.LeanToolError(
            f"Could not open declaration file `{abs_path}` for `{symbol}`."
        )

    file_content = server.get_file_contents(abs_path)

    return DeclarationInfo(
        file_path=policy.display_path(abs_path),
        content=file_content,
    )


@server.mcp.tool(
    "lean_references",
    annotations=ToolAnnotations(
        title="Find References",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def references(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int, Field(description="Column at START of identifier (1-indexed)", ge=1)
    ],
) -> ReferencesResult:
    """Find all references to a symbol (including the declaration). Position cursor at the symbol."""
    try:
        with server.lsp_client_for_file(ctx, file_path) as lsp:
            client = lsp.client
            rel_path = lsp.rel_path
            client.get_diagnostics(rel_path)

            try:
                raw_refs = client.get_references(
                    rel_path, line - 1, column - 1, include_declaration=True
                )
            except Exception as e:
                raise server.LeanToolError(f"Failed to get references: {e}")

            if raw_refs is None:
                raw_refs = []

            items: List[ReferenceLocation] = []
            policy = lsp.path_policy
            for ref in raw_refs:
                uri = ref.get("uri", "")
                r = ref.get("range", {})
                abs_path = ""
                if uri:
                    try:
                        abs_path = policy.display_path(client._uri_to_abs(uri))
                    except ValueError:
                        continue
                items.append(
                    ReferenceLocation(
                        file_path=abs_path,
                        line=r.get("start", {}).get("line", 0) + 1,
                        column=r.get("start", {}).get("character", 0) + 1,
                        end_line=r.get("end", {}).get("line", 0) + 1,
                        end_column=r.get("end", {}).get("character", 0) + 1,
                    )
                )

            return ReferencesResult(items=items)
    except server.InvalidLeanFilePathError:
        raise server.LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )
