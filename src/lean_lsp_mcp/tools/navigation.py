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
from lean_lsp_mcp.lean_names import LEAN_ID_REST
from lean_lsp_mcp.tool_registry import tool
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
)


_SOURCE_DECLARATION_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
    r"(?:theorem|lemma|def|abbrev|opaque|axiom)\s+([^\s:]+)"
)
_NAMESPACE_RE = re.compile(r"^namespace\s+(\S+)")
_SECTION_RE = re.compile(r"^section(?:\s+(\S+))?")


def _source_decl_name(lines: list[str], start_line: int) -> tuple[str, int] | None:
    """Find the declaration beginning at or just after an LSP target range."""
    if 0 <= start_line < len(lines) and "@[" in lines[start_line]:
        high = min(len(lines), start_line + 40)
        for line_number in range(start_line, high):
            match = _SOURCE_DECLARATION_RE.match(lines[line_number].strip())
            if match:
                return match.group(1), line_number

    low = max(0, start_line - 2)
    high = min(len(lines), start_line + 12)
    matches = []
    for line_number in range(low, high):
        match = _SOURCE_DECLARATION_RE.match(lines[line_number].strip())
        if match:
            matches.append((match.group(1), line_number))
    if not matches:
        return None
    return min(matches, key=lambda match: abs(match[1] - start_line))


def _to_additive_attribute(lines: list[str], declaration_line: int) -> str | None:
    """Return the nearby to_additive attribute attached to a declaration."""
    lower = max(0, declaration_line - 40)
    for attribute_line in range(declaration_line, lower - 1, -1):
        if "@[" not in lines[attribute_line]:
            continue
        if any(
            _SOURCE_DECLARATION_RE.match(lines[index].strip())
            for index in range(attribute_line + 1, declaration_line)
        ):
            break
        attribute = "\n".join(lines[attribute_line : declaration_line + 1])
        if re.search(r"@\[\s*to_additive\b", attribute):
            return attribute
    return None


def _namespace_at(lines: list[str], declaration_line: int) -> str:
    """Get the namespace prefix active at a source declaration."""
    scopes: list[tuple[str, str]] = []
    for line in lines[:declaration_line]:
        text = line.strip()
        if match := _NAMESPACE_RE.match(text):
            scopes.append(("namespace", match.group(1)))
        elif match := _SECTION_RE.match(text):
            scopes.append(("section", match.group(1) or ""))
        elif text == "end" or text.startswith("end "):
            ending = text[4:].strip() if text.startswith("end ") else ""
            if not ending and scopes:
                scopes.pop()
            else:
                for index in range(len(scopes) - 1, -1, -1):
                    if scopes[index][1] == ending:
                        del scopes[index:]
                        break
    return ".".join(name for scope, name in scopes if scope == "namespace")


def _to_additive_source_declaration(
    lines: list[str], symbol: str, start_line: int | None
) -> str | None:
    """Name the source declaration when a target is clearly to_additive output."""
    if start_line is None:
        return None
    declaration = _source_decl_name(lines, start_line)
    if declaration is None:
        return None
    declaration_name, declaration_line = declaration
    attribute = _to_additive_attribute(lines, declaration_line)
    if attribute is None:
        return None

    target_name = symbol.rsplit(".", 1)[-1]
    source_name = declaration_name.rsplit(".", 1)[-1]
    explicit_target = re.search(
        r"\bto_additive(?:\s+\([^)]*\))*\s+([A-Za-z_][\w'.]*)", attribute
    )
    if explicit_target:
        generated_name = explicit_target.group(1).rsplit(".", 1)[-1]
    elif source_name.startswith("prod_"):
        # `prod` -> `sum` is the naming rule used by Lean for this common
        # Mathlib-generated additive counterpart.
        generated_name = f"sum_{source_name[5:]}"
    else:
        return None
    if target_name != generated_name or target_name == source_name:
        return None

    namespace = _namespace_at(lines, declaration_line)
    source_name = declaration_name.removeprefix("_root_.")
    if namespace and not source_name.startswith(f"{namespace}."):
        source_name = f"{namespace}.{source_name}"
    return source_name


def _is_completion_name_char(char: str) -> bool:
    return char == "." or re.fullmatch(LEAN_ID_REST, char) is not None


def _completion_position(content: str, line: int, column: int) -> tuple[int, str]:
    lines = content.splitlines()
    if line < 1 or line > len(lines):
        return max(column - 1, 0), ""

    line_text = lines[line - 1]
    offset = min(max(column - 1, 0), len(line_text))
    # LSP positions are between characters. Callers often give the 1-indexed
    # column of the identifier's last character instead of the following one.
    if (
        offset < len(line_text)
        and _is_completion_name_char(line_text[offset])
        and (
            offset + 1 == len(line_text)
            or not _is_completion_name_char(line_text[offset + 1])
        )
    ):
        offset += 1

    start = offset
    while start > 0 and _is_completion_name_char(line_text[start - 1]):
        start -= 1
    return offset, line_text[start:offset]


def _sort_completions(
    completions: list[dict], content: str, line: int, column: int
) -> None:
    if any("sortText" in completion for completion in completions):
        completions.sort(key=lambda completion: completion.get("sortText", "\uffff"))
        return

    _, raw_prefix = _completion_position(content, line, column)
    prefix = raw_prefix.rsplit(".", 1)[-1].lower()
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
    completion_column, prefix = _completion_position(content, line, column)
    raw_completions = await client.completions(rel_path, line - 1, completion_column)
    raw_completions = [c for c in raw_completions if "label" in c]

    if not raw_completions:
        return CompletionsResult(items=[], prefix=prefix)

    _sort_completions(raw_completions, content, line, column)
    raw_completions = raw_completions[:max_completions]
    raw_completions = await _resolve_completion_details(
        client, raw_completions, resolve_details
    )
    return CompletionsResult(items=_completion_items(raw_completions), prefix=prefix)


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
    """Get declaration source with the source-line range and surrounding context.

    Generated declarations include their source declaration when it is clear.
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
    target_range = decl.get("targetRange") or decl.get("range") or {}
    target_start = target_range.get("start") or {}
    target_end = target_range.get("end") or {}
    raw_decl_start_index = target_start.get("line")
    decl_end_index = target_end.get("line")
    declaration_end_known = decl_end_index is not None
    if (
        raw_decl_start_index is not None
        and decl_end_index is not None
        and target_end.get("character") == 0
        and decl_end_index > raw_decl_start_index
    ):
        decl_end_index -= 1

    source_position = (
        _source_decl_name(file_lines, raw_decl_start_index)
        if raw_decl_start_index is not None
        else None
    )
    source_declaration = _to_additive_source_declaration(
        file_lines, symbol, raw_decl_start_index
    )
    decl_start_index = raw_decl_start_index
    if (
        source_position is not None
        and raw_decl_start_index is not None
        and source_position[1] > raw_decl_start_index
        and "@[" in "\n".join(file_lines[raw_decl_start_index : source_position[1] + 1])
    ):
        # Lean may map a generated declaration to only its `@[to_additive]`
        # attribute. Anchor the result on the actual theorem/def line instead.
        decl_start_index = source_position[1]
        if decl_end_index is None or decl_end_index < decl_start_index:
            decl_end_index = decl_start_index
            declaration_end_known = False

    declaration_start_line = (
        decl_start_index + 1 if decl_start_index is not None else None
    )
    declaration_end_line = (
        decl_end_index + 1
        if declaration_end_known and decl_end_index is not None
        else None
    )

    if full_file:
        return DeclarationInfo(
            file_path=policy.display_path(abs_path),
            content=file_content,
            start_line=1,
            end_line=total_lines,
            total_lines=total_lines,
            declaration_start_line=declaration_start_line,
            declaration_end_line=declaration_end_line,
            source_declaration=source_declaration,
        )

    decl_start = decl_start_index if decl_start_index is not None else 0
    decl_end = decl_end_index if decl_end_index is not None else decl_start
    start = max(0, decl_start - context_lines)
    end = min(total_lines, decl_end + 1 + context_lines)

    return DeclarationInfo(
        file_path=policy.display_path(abs_path),
        content="\n".join(file_lines[start:end]),
        start_line=start + 1,
        end_line=end,
        total_lines=total_lines,
        declaration_start_line=declaration_start_line,
        declaration_end_line=declaration_end_line,
        source_declaration=source_declaration,
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
