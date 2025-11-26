import asyncio
import os
import re
import time
from typing import Annotated, List, Optional, Dict
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import urllib
import orjson
import functools
import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger, configure_logging
from mcp.server.auth.settings import AuthSettings
from mcp.types import ToolAnnotations
from leanclient import LeanLSPClient, DocumentContentChange

from lean_lsp_mcp.client_utils import (
    setup_client_for_file,
    startup_client,
    infer_project_path,
)
from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.loogle import LoogleManager, loogle_remote
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.utils import (
    OutputCapture,
    deprecated,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_goal,
    get_declaration_range,
    OptionalTokenVerifier,
)


# LSP SymbolKind enum (https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
SYMBOL_KIND: Dict[int, str] = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant", 15: "string",
    16: "number", 17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum_member", 23: "struct", 24: "event", 25: "operator",
    26: "type_parameter",
}


def symbol_kind_name(kind: int | str) -> str:
    """Convert LSP SymbolKind int to readable string."""
    if isinstance(kind, str):
        return kind
    return SYMBOL_KIND.get(kind, f"unknown({kind})")


# Pydantic models for structured tool outputs
class LocalSearchResult(BaseModel):
    """A declaration found in local workspace search."""
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (theorem, def, class, etc.)")
    file: str = Field(description="Relative file path")


class LeanSearchResult(BaseModel):
    """Result from leansearch.net."""
    name: str = Field(description="Full qualified name")
    module_name: str = Field(description="Module where declared")
    kind: Optional[str] = Field(None, description="Declaration kind")
    type: Optional[str] = Field(None, description="Type signature")


class LoogleResult(BaseModel):
    """Result from loogle.lean-lang.org."""
    name: str = Field(description="Declaration name")
    type: str = Field(description="Type signature")
    module: str = Field(description="Module where declared")


class LeanFinderResult(BaseModel):
    """Result from Lean Finder semantic search."""
    full_name: str = Field(description="Full qualified name")
    formal_statement: str = Field(description="Lean type signature")
    informal_statement: str = Field(description="Natural language description")


class StateSearchResult(BaseModel):
    """Result from premise-search.com state search."""
    name: str = Field(description="Theorem/lemma name")
    score: Optional[float] = Field(None, description="Relevance score")


class PremiseResult(BaseModel):
    """A premise suggestion from hammer search."""
    name: str = Field(description="Premise name to use with simp/omega/etc.")


# LSP Diagnostic severity mapping
DIAGNOSTIC_SEVERITY: Dict[int, str] = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}


class DiagnosticMessage(BaseModel):
    """A compiler diagnostic from Lean."""
    severity: str = Field(description="Severity level: error, warning, info, or hint")
    message: str = Field(description="Diagnostic message text")
    start_line: int = Field(description="Start line number (1-indexed)")
    start_column: int = Field(description="Start column number (1-indexed)")
    end_line: int = Field(description="End line number (1-indexed)")
    end_column: int = Field(description="End column number (1-indexed)")


class GoalState(BaseModel):
    """Proof goal state at a position in a Lean file."""
    line_context: str = Field(description="The source line where goals were queried")
    goals_before: Optional[str] = Field(None, description="Goal state at line start (before tactics)")
    goals_after: Optional[str] = Field(None, description="Goal state at line end (after tactics)")
    goals: Optional[str] = Field(None, description="Goal state at specific column (when column provided)")


class CompletionItem(BaseModel):
    """A code completion suggestion."""
    label: str = Field(description="Completion text to insert")
    kind: Optional[str] = Field(None, description="Completion kind (function, variable, etc.)")
    detail: Optional[str] = Field(None, description="Additional detail about the completion")


class HoverInfo(BaseModel):
    """Hover information for a symbol."""
    symbol: str = Field(description="The symbol being hovered")
    info: str = Field(description="Type signature and documentation")
    diagnostics: List[DiagnosticMessage] = Field(default_factory=list, description="Related diagnostics at this position")


class TermGoalState(BaseModel):
    """Expected type (term goal) at a position."""
    line_context: str = Field(description="The source line where the term goal was queried")
    expected_type: Optional[str] = Field(None, description="The expected type at this position")


class OutlineEntry(BaseModel):
    """A declaration in the file outline."""
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (Thm, Def, Class, Struct, Ns, Ex)")
    start_line: int = Field(description="Start line number (1-indexed)")
    end_line: int = Field(description="End line number (1-indexed)")
    type_signature: Optional[str] = Field(None, description="Type signature if available")
    children: List["OutlineEntry"] = Field(default_factory=list, description="Nested declarations")


class FileOutline(BaseModel):
    """Structured outline of a Lean file."""
    imports: List[str] = Field(default_factory=list, description="Import statements")
    declarations: List[OutlineEntry] = Field(default_factory=list, description="Top-level declarations")


class AttemptResult(BaseModel):
    """Result of trying a code snippet."""
    snippet: str = Field(description="The code snippet that was tried")
    goal_state: Optional[str] = Field(None, description="Goal state after applying the snippet")
    diagnostics: List[DiagnosticMessage] = Field(default_factory=list, description="Diagnostics for this attempt")


class BuildResult(BaseModel):
    """Result of building the Lean project."""
    success: bool = Field(description="Whether the build succeeded")
    output: str = Field(description="Build output")
    errors: List[str] = Field(default_factory=list, description="Build errors if any")


class RunResult(BaseModel):
    """Result of running a code snippet."""
    success: bool = Field(description="Whether the code compiled successfully")
    diagnostics: List[DiagnosticMessage] = Field(default_factory=list, description="Compiler diagnostics")


class DeclarationInfo(BaseModel):
    """Information about a symbol's declaration."""
    symbol: str = Field(description="The symbol that was looked up")
    file_path: str = Field(description="Path to the file containing the declaration")
    content: str = Field(description="Content of the declaration file")


class LeanToolError(Exception):
    """Error during Lean tool execution."""
    pass


_LOG_LEVEL = os.environ.get("LEAN_LOG_LEVEL", "INFO")
configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)
logger = get_logger(__name__)


_RG_AVAILABLE, _RG_MESSAGE = check_ripgrep_status()


# Server and context
@dataclass
class AppContext:
    lean_project_path: Path | None
    client: LeanLSPClient | None
    rate_limit: Dict[str, List[int]]
    lean_search_available: bool
    loogle_manager: LoogleManager | None = None
    loogle_local_available: bool = False


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    loogle_manager: LoogleManager | None = None
    loogle_local_available = False

    try:
        lean_project_path_str = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        if not lean_project_path_str:
            lean_project_path = None
        else:
            lean_project_path = Path(lean_project_path_str).resolve()

        # Initialize local loogle if enabled via env var or CLI
        if os.environ.get("LEAN_LOOGLE_LOCAL", "").lower() in ("1", "true", "yes"):
            logger.info("Local loogle enabled, initializing...")
            loogle_manager = LoogleManager()
            if loogle_manager.ensure_installed():
                if await loogle_manager.start():
                    loogle_local_available = True
                    logger.info("Local loogle started successfully")
                else:
                    logger.warning("Local loogle failed to start, will use remote API")
            else:
                logger.warning("Local loogle installation failed, will use remote API")

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "leanfinder": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            lean_search_available=_RG_AVAILABLE,
            loogle_manager=loogle_manager,
            loogle_local_available=loogle_local_available,
        )
        yield context
    finally:
        logger.info("Closing Lean LSP client")

        if context.client:
            context.client.close()

        if loogle_manager:
            await loogle_manager.stop()


mcp_kwargs = dict(
    name="Lean LSP",
    instructions=INSTRUCTIONS,
    dependencies=["leanclient"],
    lifespan=app_lifespan,
)

auth_token = os.environ.get("LEAN_LSP_MCP_TOKEN")
if auth_token:
    mcp_kwargs["auth"] = AuthSettings(
        type="optional",
        issuer_url="http://localhost/dummy-issuer",
        resource_server_url="http://localhost/dummy-resource",
    )
    mcp_kwargs["token_verifier"] = OptionalTokenVerifier(auth_token)

mcp = FastMCP(**mcp_kwargs)


# Rate limiting: n requests per m seconds
def rate_limited(category: str, max_requests: int, per_seconds: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise KeyError(
                        "rate_limited wrapper requires ctx as a keyword argument or the first positional argument"
                    )
                ctx = args[0]
            rate_limit = ctx.request_context.lifespan_context.rate_limit
            current_time = int(time.time())
            rate_limit[category] = [
                timestamp
                for timestamp in rate_limit[category]
                if timestamp > current_time - per_seconds
            ]
            if len(rate_limit[category]) >= max_requests:
                return f"Tool limit exceeded: {max_requests} requests per {per_seconds} s. Try again later."
            rate_limit[category].append(current_time)
            return func(*args, **kwargs)

        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. " + wrapper.__doc__
        return wrapper

    return decorator


# Project level tools
@mcp.tool(
    "lean_build",
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True),
)
async def lsp_build(
    ctx: Context,
    lean_project_path: Annotated[
        Optional[str],
        Field(description="Path to Lean project. If not provided, inferred from previous tool calls."),
    ] = None,
    clean: Annotated[
        bool,
        Field(description="Run `lake clean` before building. Only use if really necessary - very slow!"),
    ] = False,
) -> BuildResult:
    """Build the Lean project and restart the LSP Server.

    Use only if needed (e.g. new imports).
    """
    if not lean_project_path:
        lean_project_path_obj = ctx.request_context.lifespan_context.lean_project_path
    else:
        lean_project_path_obj = Path(lean_project_path).resolve()
        ctx.request_context.lifespan_context.lean_project_path = lean_project_path_obj

    if lean_project_path_obj is None:
        raise LeanToolError("Lean project path not known yet. Provide `lean_project_path` explicitly or call another tool first.")

    output_lines: List[str] = []
    errors: List[str] = []

    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        if client:
            ctx.request_context.lifespan_context.client = None
            client.close()

        if clean:
            subprocess.run(["lake", "clean"], cwd=lean_project_path_obj, check=False)
            logger.info("Ran `lake clean`")

        # Fetch cache
        subprocess.run(
            ["lake", "exe", "cache", "get"], cwd=lean_project_path_obj, check=False
        )

        # Run build with progress reporting
        process = await asyncio.create_subprocess_exec(
            "lake",
            "build",
            "--verbose",
            cwd=lean_project_path_obj,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_str = line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line_str)

            # Collect error lines
            if "error" in line_str.lower():
                errors.append(line_str)

            # Parse progress: look for pattern like "[2/8]" or "[10/100]"
            match = re.search(r"\[(\d+)/(\d+)\]", line_str)
            if match:
                current_job = int(match.group(1))
                total_jobs = int(match.group(2))

                # Extract what's being built
                desc_match = re.search(
                    r"\[\d+/\d+\]\s+(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str
                )
                description = desc_match.group(1) if desc_match else "Building"

                await ctx.report_progress(
                    progress=current_job, total=total_jobs, message=description
                )

        await process.wait()

        if process.returncode != 0:
            return BuildResult(
                success=False,
                output="\n".join(output_lines),
                errors=errors or [f"Build failed with return code {process.returncode}"],
            )

        # Start LSP client (without initial build since we just did it)
        with OutputCapture():
            client = LeanLSPClient(
                lean_project_path_obj, initial_build=False, prevent_cache_get=True
            )

        logger.info("Built project and re-started LSP client")
        ctx.request_context.lifespan_context.client = client

        return BuildResult(success=True, output="\n".join(output_lines), errors=[])

    except Exception as e:
        return BuildResult(
            success=False,
            output="\n".join(output_lines),
            errors=[str(e)],
        )


# File level tools
@mcp.tool(
    "lean_file_contents",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
@deprecated
def file_contents(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    annotate_lines: Annotated[
        bool, Field(description="Annotate lines with line numbers")
    ] = True,
) -> str:
    """DEPRECATED: Will be removed soon.

    Get the text contents of a Lean file, optionally with line numbers.

    Use sparingly (bloats context). Mainly when unsure about line numbers.
    """
    # Infer project path but do not start a client
    if file_path.endswith(".lean"):
        infer_project_path(ctx, file_path)  # Silently fails for non-project files

    try:
        data = get_file_contents(file_path)
    except FileNotFoundError:
        return (
            f"File `{file_path}` does not exist. Please check the path and try again."
        )

    if annotate_lines:
        data = data.split("\n")
        max_digits = len(str(len(data)))
        annotated = ""
        for i, line in enumerate(data):
            annotated += f"{i + 1}{' ' * (max_digits - len(str(i + 1)))}: {line}\n"
        return annotated
    else:
        return data


@mcp.tool(
    "lean_file_outline",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def file_outline(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
) -> FileOutline:
    """Get a concise outline showing imports and declarations with type signatures (theorems, defs, classes, structures).

    Highly useful and token-efficient. Slow-ish.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    data = generate_outline_data(client, rel_path)

    # Convert nested dicts to OutlineEntry models
    def to_outline_entry(d: dict) -> OutlineEntry:
        return OutlineEntry(
            name=d['name'],
            kind=d['kind'],
            start_line=d['start_line'],
            end_line=d['end_line'],
            type_signature=d.get('type_signature'),
            children=[to_outline_entry(c) for c in d.get('children', [])],
        )

    return FileOutline(
        imports=data.get('imports', []),
        declarations=[to_outline_entry(d) for d in data.get('declarations', [])],
    )


def _to_diagnostic_messages(diagnostics: List[Dict]) -> List[DiagnosticMessage]:
    """Convert raw LSP diagnostics to DiagnosticMessage models."""
    result = []
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue
        severity_int = diag.get("severity", 1)
        result.append(DiagnosticMessage(
            severity=DIAGNOSTIC_SEVERITY.get(severity_int, f"unknown({severity_int})"),
            message=diag.get("message", ""),
            start_line=r["start"]["line"] + 1,
            start_column=r["start"]["character"] + 1,
            end_line=r["end"]["line"] + 1,
            end_column=r["end"]["character"] + 1,
        ))
    return result


@mcp.tool(
    "lean_diagnostic_messages",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def diagnostic_messages(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    start_line: Annotated[
        Optional[int], Field(description="Start line (1-indexed). Filters from this line.", ge=1)
    ] = None,
    end_line: Annotated[
        Optional[int], Field(description="End line (1-indexed). Filters to this line.", ge=1)
    ] = None,
    declaration_name: Annotated[
        Optional[str],
        Field(description="Name of theorem/lemma/definition. Takes precedence over line filters. Slow."),
    ] = None,
) -> List[DiagnosticMessage]:
    """Get all diagnostic msgs (errors, warnings, infos) for a Lean file.

    Common patterns:
    - "no goals to be solved" → remove extraneous tactics
    - "unknown identifier" → check imports or use lean_local_search
    - "type mismatch" → check expected vs actual types in the message
    - "failed to synthesize instance" → add instance with `haveI` or `letI`

    Tips:
    - Call without filters first to see all issues
    - Use declaration_name to focus on one theorem (slower but precise)
    - Severity: error > warning > info > hint
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    # If declaration_name is provided, get its range and use that for filtering
    if declaration_name:
        decl_range = get_declaration_range(client, rel_path, declaration_name)
        if decl_range is None:
            raise LeanToolError(f"Declaration '{declaration_name}' not found in file.")
        start_line, end_line = decl_range

    # Convert 1-indexed to 0-indexed for leanclient
    start_line_0 = (start_line - 1) if start_line is not None else None
    end_line_0 = (end_line - 1) if end_line is not None else None

    diagnostics = client.get_diagnostics(
        rel_path,
        start_line=start_line_0,
        end_line=end_line_0,
        inactivity_timeout=15.0,
    )

    return _to_diagnostic_messages(diagnostics)


@mcp.tool(
    "lean_goal",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def goal(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int],
        Field(description="Column number (1-indexed). If omitted, shows goals at both line start and end.", ge=1),
    ] = None,
) -> GoalState:
    """Get the proof goals (proof state) at a specific location in a Lean file.

    MOST IMPORTANT TOOL for proof development! Call this often.

    Returns goals_before (line start) and goals_after (line end) showing
    how the tactic on that line transforms the proof state.

    Workflow:
    1. Find the sorry line → call lean_goal on that line
    2. Read the goal state → understand what needs to be proved
    3. Try a tactic → call lean_goal again to see progress
    4. Repeat until "no goals" (proof complete)

    Tips:
    - For `sorry`: position cursor at column 1 (before the 's')
    - No column needed for most cases - default shows both before/after
    - "no goals" means proof is complete at that point
    - Watch for hypothesis changes (new `h :` bindings)
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")

    line_context = lines[line - 1]

    if column is None:
        column_end = len(line_context)
        column_start = next(
            (i for i, c in enumerate(line_context) if not c.isspace()), 0
        )
        goal_start = client.get_goal(rel_path, line - 1, column_start)
        goal_end = client.get_goal(rel_path, line - 1, column_end)

        return GoalState(
            line_context=line_context,
            goals_before=format_goal(goal_start, None),
            goals_after=format_goal(goal_end, None),
            goals=None,
        )
    else:
        goal_result = client.get_goal(rel_path, line - 1, column - 1)
        return GoalState(
            line_context=line_context,
            goals_before=None,
            goals_after=None,
            goals=format_goal(goal_result, None),
        )


@mcp.tool(
    "lean_term_goal",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def term_goal(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int], Field(description="Column number (1-indexed). Defaults to end of line.", ge=1)
    ] = None,
) -> TermGoalState:
    """Get the expected type (term goal) at a specific location in a Lean file.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")

    line_context = lines[line - 1]
    if column is None:
        column = len(line_context)

    term_goal_result = client.get_term_goal(rel_path, line - 1, column - 1)
    expected_type = None
    if term_goal_result is not None:
        rendered = term_goal_result.get("goal")
        if rendered:
            expected_type = rendered.replace("```lean\n", "").replace("\n```", "")

    return TermGoalState(line_context=line_context, expected_type=expected_type)


@mcp.tool(
    "lean_hover_info",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def hover(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int, Field(description="Column number (1-indexed). Use the start or within the term, not the end.", ge=1)
    ],
) -> HoverInfo:
    """Get type signature and documentation for any symbol in a Lean file.

    ESSENTIAL for understanding what a function/lemma does and how to use it.

    Use cases:
    - "What's the type of this variable?" → hover on the variable
    - "What arguments does this function take?" → hover on function name
    - "Is this a theorem or definition?" → hover shows the declaration

    Tips:
    - Column should be at START of identifier, not end
    - Also returns any diagnostics at that position (errors/warnings)
    - Use after lean_local_search to get full signatures
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    file_content = client.get_file_content(rel_path)
    hover_info = client.get_hover(rel_path, line - 1, column - 1)
    if hover_info is None:
        raise LeanToolError(f"No hover information at line {line}, column {column}")

    # Get the symbol and the hover information
    h_range = hover_info.get("range")
    symbol = extract_range(file_content, h_range) or ""
    info = hover_info["contents"].get("value", "No hover information available.")
    info = info.replace("```lean\n", "").replace("\n```", "").strip()

    # Add diagnostics if available
    diagnostics = client.get_diagnostics(rel_path)
    filtered = filter_diagnostics_by_position(diagnostics, line - 1, column - 1)

    return HoverInfo(
        symbol=symbol,
        info=info,
        diagnostics=_to_diagnostic_messages(filtered),
    )


# LSP CompletionItemKind mapping
COMPLETION_KIND: Dict[int, str] = {
    1: "text", 2: "method", 3: "function", 4: "constructor", 5: "field",
    6: "variable", 7: "class", 8: "interface", 9: "module", 10: "property",
    11: "unit", 12: "value", 13: "enum", 14: "keyword", 15: "snippet",
    16: "color", 17: "file", 18: "reference", 19: "folder", 20: "enum_member",
    21: "constant", 22: "struct", 23: "event", 24: "operator", 25: "type_parameter",
}


@mcp.tool(
    "lean_completions",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def completions(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    max_completions: Annotated[
        int, Field(description="Maximum number of completions to return", ge=1)
    ] = 32,
) -> List[CompletionItem]:
    """Get IDE autocompletions at a position in a Lean file.

    Use on INCOMPLETE code to discover available identifiers:
    - After `.`: field/method access (e.g., `h.` → symm, trans, mp...)
    - Partial name: `Nat.add_` → add_comm, add_assoc, add_zero...
    - After `import `: available modules

    Workflow:
    1. Write partial identifier in file
    2. Call completions at cursor position
    3. See what's available in scope

    Note: Requires the partial text to be in the file - edit first, then complete.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    raw_completions = client.get_completions(rel_path, line - 1, column - 1)

    # Convert to CompletionItem models
    items = []
    for c in raw_completions:
        if "label" not in c:
            continue
        kind_int = c.get("kind")
        kind_str = COMPLETION_KIND.get(kind_int) if kind_int else None
        items.append(CompletionItem(
            label=c["label"],
            kind=kind_str,
            detail=c.get("detail"),
        ))

    if not items:
        return []

    # Find the sort term: The last word/identifier before the cursor
    lines = content.splitlines()
    prefix = ""
    if 0 < line <= len(lines):
        text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
        if not text_before_cursor.endswith("."):
            prefix = re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[-1].lower()

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
    return items[:max_completions]


@mcp.tool(
    "lean_declaration_file",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def declaration_file(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    symbol: Annotated[str, Field(description="Symbol to look up (case sensitive). Must be present in the file!")],
) -> DeclarationInfo:
    """Get the file contents where a symbol/lemma/class/structure is declared.

    Note:
        Symbol must be present in the file! Add if necessary!
        Lean files can be large, use `lean_hover_info` before this tool.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    orig_file_content = client.get_file_content(rel_path)

    # Find the first occurence of the symbol (line and column) in the file
    position = find_start_position(orig_file_content, symbol)
    if not position:
        raise LeanToolError(f"Symbol `{symbol}` (case sensitive) not found in file. Add it first.")

    declaration = client.get_declarations(
        rel_path, position["line"], position["column"]
    )

    if len(declaration) == 0:
        raise LeanToolError(f"No declaration available for `{symbol}`.")

    # Load the declaration file
    decl = declaration[0]
    uri = decl.get("targetUri") or decl.get("uri")

    abs_path = client._uri_to_abs(uri)
    if not os.path.exists(abs_path):
        raise LeanToolError(f"Could not open declaration file `{abs_path}` for `{symbol}`.")

    file_content = get_file_contents(abs_path)

    return DeclarationInfo(symbol=symbol, file_path=str(abs_path), content=file_content)


@mcp.tool(
    "lean_multi_attempt",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def multi_attempt(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    snippets: Annotated[List[str], Field(description="List of code snippets to try (3+ recommended)")],
) -> List[AttemptResult]:
    """Try multiple tactics at a line WITHOUT modifying the file.

    BEST FOR: "Which tactic works here?" - test several approaches at once.

    Example:
        snippets=["simp", "ring", "omega", "exact?", "apply?"]

    Returns for each snippet:
    - goal_state: resulting proof state (null if error)
    - diagnostics: any errors/warnings

    Use this when unsure which tactic to try - faster than editing file repeatedly.
    Single-line snippets only. Include proper indentation.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    try:
        client.open_file(rel_path)

        results = []
        # Avoid mutating caller-provided snippets; normalize locally per attempt
        for snippet in snippets:
            snippet_str = snippet.rstrip("\n")
            payload = f"{snippet_str}\n"
            # Create a DocumentContentChange for the snippet
            change = DocumentContentChange(
                payload,
                [line - 1, 0],
                [line, 0],
            )
            # Apply the change to the file, capture diagnostics and goal state
            client.update_file(rel_path, [change])
            diag = client.get_diagnostics(rel_path)
            filtered_diag = filter_diagnostics_by_position(diag, line - 1, None)
            # Use the snippet text length without any trailing newline for the column
            goal_result = client.get_goal(rel_path, line - 1, len(snippet_str))
            goal_state = format_goal(goal_result, None)
            results.append(AttemptResult(
                snippet=snippet_str,
                goal_state=goal_state,
                diagnostics=_to_diagnostic_messages(filtered_diag),
            ))

        return results
    finally:
        try:
            client.close_files([rel_path])
        except Exception as exc:  # pragma: no cover - close failures only logged
            logger.warning(
                "Failed to close `%s` after multi_attempt: %s", rel_path, exc
            )


@mcp.tool(
    "lean_run_code",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def run_code(
    ctx: Context,
    code: Annotated[str, Field(description="Complete, self-contained Lean code snippet with all imports")],
) -> RunResult:
    """Run a complete, self-contained code snippet and return diagnostics.

    Has to include all imports and definitions!
    Only use for testing outside open files! Keep the user in the loop by editing files instead.
    """
    lifespan_context = ctx.request_context.lifespan_context
    lean_project_path = lifespan_context.lean_project_path
    if lean_project_path is None:
        raise LeanToolError("No valid Lean project path found. Run another tool first to set it up.")

    # Use a unique snippet filename to avoid collisions under concurrency
    rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
    abs_path = lean_project_path / rel_path

    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        raise LeanToolError(f"Error writing code snippet: {e}")

    client: LeanLSPClient | None = lifespan_context.client
    raw_diagnostics: List[Dict] = []
    opened_file = False

    try:
        if client is None:
            startup_client(ctx)
            client = lifespan_context.client
            if client is None:
                raise LeanToolError("Failed to initialize Lean client for run_code.")

        assert client is not None
        client.open_file(rel_path)
        opened_file = True
        raw_diagnostics = client.get_diagnostics(rel_path, inactivity_timeout=15.0)
    finally:
        if opened_file:
            try:
                client.close_files([rel_path])
            except Exception as exc:
                logger.warning("Failed to close `%s` after run_code: %s", rel_path, exc)
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to remove temporary Lean snippet `%s`: %s", abs_path, e)

    diagnostics = _to_diagnostic_messages(raw_diagnostics)
    has_errors = any(d.severity == "error" for d in diagnostics)

    return RunResult(success=not has_errors, diagnostics=diagnostics)


class LocalSearchError(Exception):
    """Error during local search."""
    pass


def _build_declaration_index(project_root: Path, limit: int = 5000) -> List[LocalSearchResult]:
    """Build a declaration index for a project by searching common prefixes."""
    if not _RG_AVAILABLE:
        return []

    all_results: Dict[str, LocalSearchResult] = {}

    # Search with empty prefix to get all declarations (ripgrep pattern matches any declaration)
    try:
        raw_results = lean_local_search(query="", limit=limit, project_root=project_root)
        for r in raw_results:
            key = f"{r['name']}:{r['file']}"
            if key not in all_results:
                all_results[key] = LocalSearchResult(
                    name=r["name"],
                    kind=r["kind"],
                    file=r["file"]
                )
    except RuntimeError:
        pass

    return list(all_results.values())


@mcp.tool(
    "lean_local_search",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def local_search(
    ctx: Context,
    query: Annotated[str, Field(description="Declaration name or prefix to search for")],
    limit: Annotated[int, Field(description="Max matches to return", ge=1)] = 10,
    project_root: Annotated[
        Optional[str], Field(description="Lean project root. Inferred if not provided.")
    ] = None,
) -> List[LocalSearchResult]:
    """Confirm declarations exist in the current workspace to prevent hallucinating APIs.

    FASTEST search tool - use BEFORE trying a lemma name!
    Searches: theorems, lemmas, defs, classes, instances, structures, inductives, abbrevs.

    Use cases:
    - "Does `Nat.add_comm` exist?" → search "Nat.add_comm"
    - "What's in List module?" → search "List."
    - "Find mul lemmas" → search "mul_"

    Returns kind (theorem/def/class/etc.) and file path.
    Use lean_hover_info after to get the full type signature.
    """
    if not _RG_AVAILABLE:
        raise LocalSearchError(_RG_MESSAGE)

    lifespan = ctx.request_context.lifespan_context
    stored_root = lifespan.lean_project_path

    if project_root:
        try:
            resolved_root = Path(project_root).expanduser().resolve()
        except OSError as exc:
            raise LocalSearchError(f"Invalid project root '{project_root}': {exc}")
        if not resolved_root.exists():
            raise LocalSearchError(f"Project root '{project_root}' does not exist.")
        lifespan.lean_project_path = resolved_root
    else:
        resolved_root = stored_root

    if resolved_root is None:
        raise LocalSearchError("Lean project path not set. Call a file-based tool first.")

    try:
        raw_results = lean_local_search(
            query=query.strip(), limit=limit, project_root=resolved_root
        )
        return [
            LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
            for r in raw_results
        ]
    except RuntimeError as exc:
        raise LocalSearchError(f"Search failed: {exc}")


@mcp.tool(
    "lean_leansearch",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(
    ctx: Context,
    query: Annotated[str, Field(description="Natural language or Lean term search query")],
    num_results: Annotated[int, Field(description="Max results to return", ge=1)] = 5,
) -> List[LeanSearchResult]:
    """Search Mathlib theorems using natural language via leansearch.net.

    BEST FOR: "I need a lemma that says X" in plain English.
    Rate limited: 3 requests per 30 seconds.

    Query examples:
    - "sum of two even numbers is even"
    - "injective function has left inverse"
    - "Cauchy Schwarz inequality"
    - Lean term: "{f : A → B} (hf : Injective f) : ∃ g, LeftInverse g f"

    Tip: Use lean_local_search first to verify returned names exist in your project.
    """
    headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
    payload = orjson.dumps({"num_results": str(num_results), "query": [query]})

    req = urllib.request.Request(
        "https://leansearch.net/search",
        data=payload,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        results = orjson.loads(response.read())

    if not results or not results[0]:
        return []

    raw_results = [r["result"] for r in results[0][:num_results]]
    return [
        LeanSearchResult(
            name=".".join(r["name"]),
            module_name=".".join(r["module_name"]),
            kind=r.get("kind"),
            type=r.get("type"),
        )
        for r in raw_results
    ]


@mcp.tool(
    "lean_loogle",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def loogle(
    ctx: Context,
    query: Annotated[str, Field(description="Loogle query pattern (constant, name, type shape, etc.)")],
    num_results: Annotated[int, Field(description="Max results to return", ge=1)] = 8,
) -> List[LoogleResult]:
    """Search Mathlib by TYPE SIGNATURE pattern via loogle.lean-lang.org.

    BEST FOR: "Find lemma with this type shape" or "uses this constant".
    Rate limited: 3 requests per 30 seconds.

    Query patterns:
    - Constant: `Real.sin` → lemmas mentioning Real.sin
    - Name substring: `"comm"` → lemmas with "comm" in name
    - Type shape: `(?a → ?b) → List ?a → List ?b` → finds map
    - Subexpression: `_ * (_ ^ _)` → products with powers
    - Conclusion: `|- _ + _ = _ + _` → addition equations

    Returns full type signatures - very useful for understanding how to apply lemmas.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Try local loogle first if available (no rate limiting)
    if app_ctx.loogle_local_available and app_ctx.loogle_manager:
        try:
            results = await app_ctx.loogle_manager.query(query, num_results)
            for result in results:
                result.pop("doc", None)
            return results if results else "No results found."
        except Exception as e:
            logger.warning(f"Local loogle failed: {e}, falling back to remote")

    # Fall back to remote (with rate limiting)
    rate_limit = app_ctx.rate_limit["loogle"]
    now = int(time.time())
    rate_limit[:] = [t for t in rate_limit if now - t < 30]
    if len(rate_limit) >= 3:
        return "Rate limit exceeded: 3 requests per 30s. Use --loogle-local to avoid limits."
    rate_limit.append(now)

    return loogle_remote(query, num_results)


@mcp.tool(
    "lean_leanfinder",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
@rate_limited("leanfinder", max_requests=10, per_seconds=30)
def leanfinder(
    ctx: Context,
    query: Annotated[str, Field(description="Mathematical concept, proof state, or statement to search")],
    num_results: Annotated[int, Field(description="Max results to return", ge=1)] = 5,
) -> List[LeanFinderResult]:
    """Semantic search of Mathlib by mathematical MEANING via Lean Finder.

    BEST FOR: When you know what you want mathematically but not the exact name.
    Rate limited: 10 requests per 30 seconds (highest limit).

    Query strategies:
    - Mathematical statement: "commutativity of addition on natural numbers"
    - Proof state + goal: "I have h : n < m and need to show n + 1 < m + 1"
    - Informal question: "how to prove a function is continuous"

    Returns informal_statement (English) + formal_statement (Lean) for each result.
    Use multiple focused queries rather than one complex query.
    """
    headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
    request_url = (
        "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
    )
    payload = orjson.dumps({"inputs": query, "top_k": int(num_results)})
    req = urllib.request.Request(
        request_url, data=payload, headers=headers, method="POST"
    )

    results = []
    with urllib.request.urlopen(req, timeout=30) as response:
        data = orjson.loads(response.read())
        for result in data["results"]:
            if (
                "https://leanprover-community.github.io/mathlib4_docs"
                not in result["url"]
            ):  # Only include mathlib4 results
                continue
            match = re.search(r"pattern=(.*?)#doc", result["url"])
            if match:
                results.append(LeanFinderResult(
                    full_name=match.group(1),
                    formal_statement=result["formal_statement"],
                    informal_statement=result["informal_statement"],
                ))

    return results


@mcp.tool(
    "lean_state_search",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results to return", ge=1)] = 5,
) -> List[StateSearchResult]:
    """Find lemmas that can close the CURRENT GOAL at a position.

    BEST FOR: "What lemma finishes this proof state?"
    Rate limited: 3 requests per 30 seconds.

    Automatically reads the goal at the given position and searches premise-search.com.
    Returns lemma names ranked by relevance score.

    Workflow:
    1. Position at a sorry or tactic
    2. Call state_search with that position
    3. Try returned lemmas with `exact`, `apply`, or as simp hints
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise ValueError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise ValueError(f"No goals found at line {line}, column {column}")

    goal_str = urllib.parse.quote(goal["goals"][0])

    url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
    req = urllib.request.Request(
        f"{url}/api/search?query={goal_str}&results={num_results}&rev=v4.22.0",
        headers={"User-Agent": "lean-lsp-mcp/0.1"},
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        results = orjson.loads(response.read())

    return [
        StateSearchResult(name=r["name"], score=r.get("score"))
        for r in results
    ]


@mcp.tool(
    "lean_hammer_premise",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results to return", ge=1)] = 32,
) -> List[PremiseResult]:
    """Get premises for automation tactics (simp, omega, aesop) at a position.

    BEST FOR: "What lemmas should I feed to simp/omega/decide?"
    Rate limited: 3 requests per 30 seconds.

    Returns premise names optimized for use with:
    - `simp only [premise1, premise2, ...]`
    - `omega` / `decide` (for decidable goals)
    - `aesop (add unsafe [premise1, premise2])`

    Higher num_results (32+) often helps automation find the right combination.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise ValueError("Invalid Lean file path: Unable to start LSP server or load file")

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise ValueError(f"No goals found at line {line}, column {column}")

    data = {
        "state": goal["goals"][0],
        "new_premises": [],
        "k": num_results,
    }

    url = os.getenv("LEAN_HAMMER_URL", "http://leanpremise.net")
    req = urllib.request.Request(
        url + "/retrieve",
        headers={
            "User-Agent": "lean-lsp-mcp/0.1",
            "Content-Type": "application/json",
        },
        method="POST",
        data=orjson.dumps(data),
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        results = orjson.loads(response.read())

    return [PremiseResult(name=r["name"]) for r in results]


if __name__ == "__main__":
    mcp.run()
