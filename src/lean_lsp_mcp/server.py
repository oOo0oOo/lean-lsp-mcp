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
from lean_lsp_mcp.models import (
    LocalSearchResult,
    LeanSearchResult,
    LoogleResult,
    LeanFinderResult,
    StateSearchResult,
    PremiseResult,
    DiagnosticMessage,
    GoalState,
    CompletionItem,
    HoverInfo,
    TermGoalState,
    FileOutline,
    AttemptResult,
    BuildResult,
    RunResult,
    DeclarationInfo,
)
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    OutputCapture,
    deprecated,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_goal,
    get_declaration_range,
    OptionalTokenVerifier,
)

# LSP Diagnostic severity: 1=error, 2=warning, 3=info, 4=hint
DIAGNOSTIC_SEVERITY: Dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}


class LeanToolError(Exception):
    pass


def _to_json_array(items: List[BaseModel]) -> str:
    """Serialize list of models as JSON array (avoids FastMCP list flattening)."""
    return orjson.dumps(
        [item.model_dump() for item in items], option=orjson.OPT_INDENT_2
    ).decode()


_LOG_LEVEL = os.environ.get("LEAN_LOG_LEVEL", "INFO")
configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)
logger = get_logger(__name__)


_RG_AVAILABLE, _RG_MESSAGE = check_ripgrep_status()


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


@mcp.tool(
    "lean_build",
    annotations=ToolAnnotations(
        title="Build Project",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def lsp_build(
    ctx: Context,
    lean_project_path: Annotated[
        Optional[str], Field(description="Path to Lean project")
    ] = None,
    clean: Annotated[bool, Field(description="Run lake clean first (slow)")] = False,
) -> BuildResult:
    """Build the Lean project and restart LSP. Use only if needed (e.g. new imports)."""
    if not lean_project_path:
        lean_project_path_obj = ctx.request_context.lifespan_context.lean_project_path
    else:
        lean_project_path_obj = Path(lean_project_path).resolve()
        ctx.request_context.lifespan_context.lean_project_path = lean_project_path_obj

    if lean_project_path_obj is None:
        raise LeanToolError(
            "Lean project path not known yet. Provide `lean_project_path` explicitly or call another tool first."
        )

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
                errors=errors
                or [f"Build failed with return code {process.returncode}"],
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


@mcp.tool(
    "lean_file_contents",
    annotations=ToolAnnotations(
        title="File Contents (Deprecated)",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
@deprecated
def file_contents(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    annotate_lines: Annotated[bool, Field(description="Add line numbers")] = True,
) -> str:
    """DEPRECATED. Get file contents with optional line numbers."""
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
    annotations=ToolAnnotations(
        title="File Outline",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def file_outline(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
) -> FileOutline:
    """Get imports and declarations with type signatures. Token-efficient."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    return generate_outline_data(client, rel_path)


def _to_diagnostic_messages(diagnostics: List[Dict]) -> List[DiagnosticMessage]:
    result = []
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue
        severity_int = diag.get("severity", 1)
        result.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(
                    severity_int, f"unknown({severity_int})"
                ),
                message=diag.get("message", ""),
                line=r["start"]["line"] + 1,
                column=r["start"]["character"] + 1,
            )
        )
    return result


@mcp.tool(
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
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    start_line: Annotated[
        Optional[int], Field(description="Filter from line", ge=1)
    ] = None,
    end_line: Annotated[
        Optional[int], Field(description="Filter to line", ge=1)
    ] = None,
    declaration_name: Annotated[
        Optional[str], Field(description="Filter to declaration (slow)")
    ] = None,
) -> str:
    """Get compiler diagnostics (errors, warnings, infos) for a Lean file."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

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

    return _to_json_array(_to_diagnostic_messages(diagnostics))


@mcp.tool(
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
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int],
        Field(description="Column (1-indexed). Omit for before/after", ge=1),
    ] = None,
) -> GoalState:
    """Get proof goals at a position. MOST IMPORTANT tool - use often!

    Omit column to see goals_before (line start) and goals_after (line end),
    showing how the tactic transforms the state. "no goals" = proof complete.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

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
        before = format_goal(goal_start, None)
        after = format_goal(goal_end, None)
        goals = f"{before} → {after}" if before != after else after
        return GoalState(line_context=line_context, goals=goals)
    else:
        goal_result = client.get_goal(rel_path, line - 1, column - 1)
        return GoalState(
            line_context=line_context, goals=format_goal(goal_result, None)
        )


@mcp.tool(
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
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        Optional[int], Field(description="Column (defaults to end of line)", ge=1)
    ] = None,
) -> TermGoalState:
    """Get the expected type at a position."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

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
    annotations=ToolAnnotations(
        title="Hover Info",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def hover(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column at START of identifier", ge=1)],
) -> HoverInfo:
    """Get type signature and docs for a symbol. Essential for understanding APIs."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

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


@mcp.tool(
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
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    max_completions: Annotated[int, Field(description="Max completions", ge=1)] = 32,
) -> str:
    """Get IDE autocompletions. Use on INCOMPLETE code (after `.` or partial name)."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    raw_completions = client.get_completions(rel_path, line - 1, column - 1)

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
        return "[]"

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
    return _to_json_array(items[:max_completions])


@mcp.tool(
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
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    symbol: Annotated[
        str, Field(description="Symbol (case sensitive, must be in file)")
    ],
) -> DeclarationInfo:
    """Get file where a symbol is declared. Symbol must be present in file first."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    orig_file_content = client.get_file_content(rel_path)

    # Find the first occurence of the symbol (line and column) in the file
    position = find_start_position(orig_file_content, symbol)
    if not position:
        raise LeanToolError(
            f"Symbol `{symbol}` (case sensitive) not found in file. Add it first."
        )

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
        raise LeanToolError(
            f"Could not open declaration file `{abs_path}` for `{symbol}`."
        )

    file_content = get_file_contents(abs_path)

    return DeclarationInfo(file_path=str(abs_path), content=file_content)


@mcp.tool(
    "lean_multi_attempt",
    annotations=ToolAnnotations(
        title="Multi-Attempt",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def multi_attempt(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    snippets: Annotated[
        List[str], Field(description="Tactics to try (3+ recommended)")
    ],
) -> str:
    """Try multiple tactics without modifying file. Returns goal state for each."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    try:
        client.open_file(rel_path)

        results: List[AttemptResult] = []
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
            results.append(
                AttemptResult(
                    snippet=snippet_str,
                    goal_state=goal_state,
                    diagnostics=_to_diagnostic_messages(filtered_diag),
                )
            )

        return _to_json_array(results)
    finally:
        try:
            client.close_files([rel_path])
        except Exception as exc:  # pragma: no cover - close failures only logged
            logger.warning(
                "Failed to close `%s` after multi_attempt: %s", rel_path, exc
            )


@mcp.tool(
    "lean_run_code",
    annotations=ToolAnnotations(
        title="Run Code",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def run_code(
    ctx: Context,
    code: Annotated[str, Field(description="Self-contained Lean code with imports")],
) -> RunResult:
    """Run a code snippet and return diagnostics. Must include all imports."""
    lifespan_context = ctx.request_context.lifespan_context
    lean_project_path = lifespan_context.lean_project_path
    if lean_project_path is None:
        raise LeanToolError(
            "No valid Lean project path found. Run another tool first to set it up."
        )

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
            logger.warning(
                "Failed to remove temporary Lean snippet `%s`: %s", abs_path, e
            )

    diagnostics = _to_diagnostic_messages(raw_diagnostics)
    has_errors = any(d.severity == "error" for d in diagnostics)

    return RunResult(success=not has_errors, diagnostics=diagnostics)


class LocalSearchError(Exception):
    pass


@mcp.tool(
    "lean_local_search",
    annotations=ToolAnnotations(
        title="Local Search",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def local_search(
    ctx: Context,
    query: Annotated[str, Field(description="Declaration name or prefix")],
    limit: Annotated[int, Field(description="Max matches", ge=1)] = 10,
    project_root: Annotated[
        Optional[str], Field(description="Project root (inferred if omitted)")
    ] = None,
) -> str:
    """Fast local search to verify declarations exist. Use BEFORE trying a lemma name."""
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
        raise LocalSearchError(
            "Lean project path not set. Call a file-based tool first."
        )

    try:
        raw_results = lean_local_search(
            query=query.strip(), limit=limit, project_root=resolved_root
        )
        results = [
            LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
            for r in raw_results
        ]
        return _to_json_array(results)
    except RuntimeError as exc:
        raise LocalSearchError(f"Search failed: {exc}")


@mcp.tool(
    "lean_leansearch",
    annotations=ToolAnnotations(
        title="LeanSearch",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(
    ctx: Context,
    query: Annotated[str, Field(description="Natural language or Lean term query")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> str:
    """Search Mathlib via leansearch.net using natural language.

    Examples: "sum of two even numbers is even", "Cauchy-Schwarz inequality",
    "{f : A → B} (hf : Injective f) : ∃ g, LeftInverse g f"
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
        return "[]"

    raw_results = [r["result"] for r in results[0][:num_results]]
    items = [
        LeanSearchResult(
            name=".".join(r["name"]),
            module_name=".".join(r["module_name"]),
            kind=r.get("kind"),
            type=r.get("type"),
        )
        for r in raw_results
    ]
    return _to_json_array(items)


@mcp.tool(
    "lean_loogle",
    annotations=ToolAnnotations(
        title="Loogle",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def loogle(
    ctx: Context,
    query: Annotated[
        str, Field(description="Type pattern, constant, or name substring")
    ],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 8,
) -> str:
    """Search Mathlib by type signature via loogle.lean-lang.org.

    Examples: `Real.sin`, `"comm"`, `(?a → ?b) → List ?a → List ?b`,
    `_ * (_ ^ _)`, `|- _ < _ → _ + 1 < _ + 1`
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Try local loogle first if available (no rate limiting)
    if app_ctx.loogle_local_available and app_ctx.loogle_manager:
        try:
            results = await app_ctx.loogle_manager.query(query, num_results)
            if not results:
                return "No results found."
            items = [
                LoogleResult(
                    name=r.get("name", ""),
                    type=r.get("type", ""),
                    module=r.get("module", ""),
                )
                for r in results
            ]
            return _to_json_array(items)
        except Exception as e:
            logger.warning(f"Local loogle failed: {e}, falling back to remote")

    # Fall back to remote (with rate limiting)
    rate_limit = app_ctx.rate_limit["loogle"]
    now = int(time.time())
    rate_limit[:] = [t for t in rate_limit if now - t < 30]
    if len(rate_limit) >= 3:
        return "Rate limit exceeded: 3 requests per 30s. Use --loogle-local to avoid limits."
    rate_limit.append(now)

    result = loogle_remote(query, num_results)
    if isinstance(result, str):
        return result  # Error message
    return _to_json_array(result)


@mcp.tool(
    "lean_leanfinder",
    annotations=ToolAnnotations(
        title="Lean Finder",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leanfinder", max_requests=10, per_seconds=30)
def leanfinder(
    ctx: Context,
    query: Annotated[str, Field(description="Mathematical concept or proof state")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> str:
    """Semantic search by mathematical meaning via Lean Finder.

    Examples: "commutativity of addition on natural numbers",
    "I have h : n < m and need n + 1 < m + 1", proof state text.
    """
    headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
    request_url = "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
    payload = orjson.dumps({"inputs": query, "top_k": int(num_results)})
    req = urllib.request.Request(
        request_url, data=payload, headers=headers, method="POST"
    )

    results: List[LeanFinderResult] = []
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
                results.append(
                    LeanFinderResult(
                        full_name=match.group(1),
                        formal_statement=result["formal_statement"],
                        informal_statement=result["informal_statement"],
                    )
                )

    return _to_json_array(results)


@mcp.tool(
    "lean_state_search",
    annotations=ToolAnnotations(
        title="State Search",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> str:
    """Find lemmas to close the goal at a position. Searches premise-search.com."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position or check if the proof is complete."
        )

    goal_str = urllib.parse.quote(goal["goals"][0])

    url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
    req = urllib.request.Request(
        f"{url}/api/search?query={goal_str}&results={num_results}&rev=v4.22.0",
        headers={"User-Agent": "lean-lsp-mcp/0.1"},
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        results = orjson.loads(response.read())

    items = [StateSearchResult(name=r["name"]) for r in results]
    return _to_json_array(items)


@mcp.tool(
    "lean_hammer_premise",
    annotations=ToolAnnotations(
        title="Hammer Premises",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 32,
) -> str:
    """Get premise suggestions for automation tactics at a goal position.

    Returns lemma names to try with `simp only [...]`, `aesop`, or as hints.
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position or check if the proof is complete."
        )

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

    items = [PremiseResult(name=r["name"]) for r in results]
    return _to_json_array(items)


if __name__ == "__main__":
    mcp.run()
