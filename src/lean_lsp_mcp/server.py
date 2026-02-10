import asyncio
import functools
import inspect
import json
import logging.config
import os
import re
import ssl
import time
import urllib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import certifi
import orjson
from leanclient import DocumentContentChange, LeanLSPClient
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp.client_utils import (
    infer_project_path,
    setup_client_for_file,
    startup_client,
)
from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.loogle import LoogleManager, loogle_remote
from lean_lsp_mcp.repl import Repl, repl_enabled
from lean_lsp_mcp.models import (
    AttemptResult,
    BuildResult,
    CompletionItem,
    CompletionsResult,
    DeclarationInfo,
    DiagnosticMessage,
    # Wrapper models for list-returning tools
    DiagnosticsResult,
    FileOutline,
    GoalState,
    HoverInfo,
    LeanFinderResult,
    LeanFinderResults,
    LeanExploreResult,
    LeanExploreResults,
    LeanSearchResult,
    LeanSearchResults,
    LocalSearchResult,
    LocalSearchResults,
    LoogleResult,
    LoogleResults,
    MultiAttemptResult,
    PremiseResult,
    PremiseResults,
    ProofProfileResult,
    RunResult,
    StateSearchResult,
    StateSearchResults,
    TermGoalState,
)

# REPL models not imported - low-level REPL tools not exposed to keep API simple.
# The model uses lean_multi_attempt which handles REPL internally.
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    LeanToolError,
    OptionalTokenVerifier,
    OutputCapture,
    check_lsp_response,
    deprecated,
    extract_failed_dependency_paths,
    extract_goals_list,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    get_declaration_range,
    is_build_stderr,
)

_LEANEXPLORE_LOCAL_ENV = "LEAN_EXPLORE_LOCAL"
_LEANEXPLORE_BACKEND_ENV = "LEAN_EXPLORE_BACKEND"

# LSP Diagnostic severity: 1=error, 2=warning, 3=info, 4=hint
DIAGNOSTIC_SEVERITY: Dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}


async def _urlopen_json(req: urllib.request.Request, timeout: float):
    """Run urllib.request.urlopen in a worker thread to avoid blocking the event loop."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _do_request():
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
            return orjson.loads(response.read())

    return await asyncio.to_thread(_do_request)


async def _safe_report_progress(
    ctx: Context, *, progress: int, total: int, message: str
) -> None:
    try:
        await ctx.report_progress(progress=progress, total=total, message=message)
    except Exception:
        return


_LOG_FILE_CONFIG = os.environ.get("LEAN_LOG_FILE_CONFIG", None)
_LOG_LEVEL = os.environ.get("LEAN_LOG_LEVEL", "INFO")
if _LOG_FILE_CONFIG:
    try:
        if _LOG_FILE_CONFIG.endswith((".yaml", ".yml")):
            import yaml

            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
        elif _LOG_FILE_CONFIG.endswith(".json"):
            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = orjson.loads(f.read())
            logging.config.dictConfig(cfg)
        else:
            # .ini / fileConfig
            logging.config.fileConfig(_LOG_FILE_CONFIG, disable_existing_loggers=False)
    except Exception as e:
        # fallback to LEAN_LOG_LEVEL so server still runs
        # use the existing configure_logging helper to set level
        configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)
        logger = get_logger(__name__)  # temporary to emit the warning
        logger.warning(
            "Failed to load logging config %s: %s. Falling back to LEAN_LOG_LEVEL.",
            _LOG_FILE_CONFIG,
            e,
        )
else:
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
    # REPL for efficient multi-attempt execution
    repl: Repl | None = None
    repl_enabled: bool = False
    leanexplore_local_enabled: bool = False
    leanexplore_service: object | None = None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    loogle_manager: LoogleManager | None = None
    loogle_local_available = False
    repl: Repl | None = None
    repl_on = False
    leanexplore_local_enabled = False
    leanexplore_service = None

    try:
        lean_project_path_str = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        if not lean_project_path_str:
            lean_project_path = None
        else:
            lean_project_path = Path(lean_project_path_str).resolve()

        # Initialize local loogle if enabled via env var or CLI
        if os.environ.get("LEAN_LOOGLE_LOCAL", "").lower() in ("1", "true", "yes"):
            logger.info("Local loogle enabled, initializing...")
            loogle_manager = LoogleManager(project_path=lean_project_path)
            if loogle_manager.ensure_installed():
                if await loogle_manager.start():
                    loogle_local_available = True
                    logger.info("Local loogle started successfully")
                else:
                    logger.warning("Local loogle failed to start, will use remote API")
            else:
                logger.warning("Local loogle installation failed, will use remote API")

        # Track LeanExplore local mode for lazy init
        leanexplore_local_enabled = _leanexplore_use_local_backend()

        # Initialize REPL if enabled
        if repl_enabled():
            if lean_project_path:
                from lean_lsp_mcp.repl import find_repl_binary

                repl_bin = find_repl_binary(str(lean_project_path))
                if repl_bin:
                    logger.info("REPL enabled, using: %s", repl_bin)
                    repl = Repl(project_dir=str(lean_project_path), repl_path=repl_bin)
                    repl_on = True
                    logger.info("REPL initialized: timeout=%ds", repl.timeout)
                else:
                    logger.warning(
                        "REPL enabled but binary not found. "
                        'Add `require repl from git "https://github.com/leanprover-community/repl"` '
                        "to lakefile and run `lake build repl`. Falling back to LSP."
                    )
            else:
                logger.warning("REPL requires LEAN_PROJECT_PATH to be set")

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            rate_limit={
                "leanexplore": [],
                "leansearch": [],
                "loogle": [],
                "leanfinder": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            lean_search_available=_RG_AVAILABLE,
            loogle_manager=loogle_manager,
            loogle_local_available=loogle_local_available,
            repl=repl,
            repl_enabled=repl_on,
            leanexplore_local_enabled=leanexplore_local_enabled,
            leanexplore_service=leanexplore_service,
        )
        yield context
    finally:
        logger.info("Closing Lean LSP client")

        if context.client:
            context.client.close()

        if loogle_manager:
            await loogle_manager.stop()

        if repl:
            await repl.close()


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
        def _apply_rate_limit(args, kwargs):
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
                return (
                    False,
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s. Try again later.",
                )
            rate_limit[category].append(current_time)
            return True, None

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return func(*args, **kwargs)

        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. {doc}"
        return wrapper

    return decorator


@mcp.tool(
    "lean_build",
    annotations=ToolAnnotations(
        title="Build Project",
        readOnlyHint=False,
        destructiveHint=True,
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
    output_lines: Annotated[
        int, Field(description="Return last N lines of build log (0=none)")
    ] = 20,
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

    log_lines: List[str] = []
    errors: List[str] = []

    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        if client:
            ctx.request_context.lifespan_context.client = None
            client.close()

        if clean:
            await ctx.report_progress(
                progress=1, total=16, message="Running `lake clean`"
            )
            clean_proc = await asyncio.create_subprocess_exec(
                "lake", "clean", cwd=lean_project_path_obj
            )
            await clean_proc.wait()

        await ctx.report_progress(
            progress=2, total=16, message="Running `lake exe cache get`"
        )
        cache_proc = await asyncio.create_subprocess_exec(
            "lake", "exe", "cache", "get", cwd=lean_project_path_obj
        )
        await cache_proc.wait()

        # Run build with progress reporting
        process = await asyncio.create_subprocess_exec(
            "lake",
            "build",
            "--verbose",
            cwd=lean_project_path_obj,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        while line := await process.stdout.readline():
            line_str = line.decode("utf-8", errors="replace").rstrip()

            if line_str.startswith("trace:") or "LEAN_PATH=" in line_str:
                continue

            log_lines.append(line_str)
            if "error" in line_str.lower():
                errors.append(line_str)

            # Parse progress: "[2/8] Building Foo (1.2s)" -> (2, 8, "Building Foo")
            if m := re.search(
                r"\[(\d+)/(\d+)\]\s*(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str
            ):
                await ctx.report_progress(
                    progress=int(m.group(1)),
                    total=int(m.group(2)),
                    message=m.group(3) or "Building",
                )

        await process.wait()

        if process.returncode != 0:
            return BuildResult(
                success=False,
                output="\n".join(log_lines[-output_lines:]) if output_lines else "",
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

        return BuildResult(
            success=True,
            output="\n".join(log_lines[-output_lines:]) if output_lines else "",
            errors=[],
        )

    except Exception as e:
        return BuildResult(
            success=False,
            output="\n".join(log_lines[-output_lines:]) if output_lines else "",
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
    """Convert LSP diagnostics to DiagnosticMessage models."""
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


def _process_diagnostics(
    diagnostics: List[Dict], build_success: bool
) -> DiagnosticsResult:
    """Process diagnostics, extracting dependency paths from build stderr.

    Args:
        diagnostics: List of diagnostic dicts from leanclient
        build_success: Whether the build succeeded (from leanclient.DiagnosticsResult.success)
    """
    items = []
    failed_deps: List[str] = []

    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue

        severity_int = diag.get("severity", 1)
        message = diag.get("message", "")
        line = r["start"]["line"] + 1
        column = r["start"]["character"] + 1

        # Check if this is a build failure at (1,1) - extract dependency paths, skip the item
        if line == 1 and column == 1 and is_build_stderr(message):
            failed_deps = extract_failed_dependency_paths(message)
            continue  # Don't include the build stderr blob as a diagnostic item

        # Normal diagnostic from the queried file
        items.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(
                    severity_int, f"unknown({severity_int})"
                ),
                message=message,
                line=line,
                column=column,
            )
        )

    return DiagnosticsResult(
        success=build_success,
        items=items,
        failed_dependencies=failed_deps,
    )


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
) -> DiagnosticsResult:
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

    result = client.get_diagnostics(
        rel_path,
        start_line=start_line_0,
        end_line=end_line_0,
        inactivity_timeout=15.0,
    )

    return _process_diagnostics(result.diagnostics, result.success)


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
        check_lsp_response(goal_start, "get_goal", allow_none=True)
        goal_end = client.get_goal(rel_path, line - 1, column_end)
        return GoalState(
            line_context=line_context,
            goals_before=extract_goals_list(goal_start),
            goals_after=extract_goals_list(goal_end),
        )
    else:
        goal_result = client.get_goal(rel_path, line - 1, column - 1)
        check_lsp_response(goal_result, "get_goal", allow_none=True)
        return GoalState(
            line_context=line_context, goals=extract_goals_list(goal_result)
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
    check_lsp_response(term_goal_result, "get_term_goal", allow_none=True)
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
    check_lsp_response(hover_info, "get_hover", allow_none=True)
    if hover_info is None:
        raise LeanToolError(f"No hover information at line {line}, column {column}")

    # Get the symbol and the hover information
    h_range = hover_info.get("range")
    symbol = extract_range(file_content, h_range) or ""
    info = hover_info["contents"].get("value", "No hover information available.")
    info = info.replace("```lean\n", "").replace("\n```", "").strip()

    # Add diagnostics if available
    diagnostics = client.get_diagnostics(rel_path)
    check_lsp_response(diagnostics, "get_diagnostics")
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
) -> CompletionsResult:
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
    return CompletionsResult(items=items[:max_completions])


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


async def _multi_attempt_repl(
    ctx: Context,
    file_path: str,
    line: int,
    snippets: List[str],
) -> MultiAttemptResult | None:
    """Try tactics using REPL (fast path)."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    if not app_ctx.repl_enabled or not app_ctx.repl:
        return None

    try:
        content = get_file_contents(file_path)
        if content is None:
            return None
        lines = content.splitlines()
        if line > len(lines):
            return None

        base_code = "\n".join(lines[: line - 1])
        repl_results = await app_ctx.repl.run_snippets(base_code, snippets)

        results = []
        for snippet, pr in zip(snippets, repl_results):
            diagnostics = [
                DiagnosticMessage(
                    severity=m.get("severity", "info"),
                    message=m.get("data", ""),
                    line=m.get("pos", {}).get("line", 0),
                    column=m.get("pos", {}).get("column", 0),
                )
                for m in (pr.messages or [])
            ]
            if pr.error:
                diagnostics.append(
                    DiagnosticMessage(
                        severity="error", message=pr.error, line=0, column=0
                    )
                )
            results.append(
                AttemptResult(
                    snippet=snippet.rstrip("\n"),
                    goals=pr.goals or [],
                    diagnostics=diagnostics,
                )
            )
        return MultiAttemptResult(items=results)
    except Exception as e:
        logger.debug(f"REPL multi_attempt failed: {e}")
        return None


def _multi_attempt_lsp(
    ctx: Context,
    file_path: str,
    line: int,
    snippets: List[str],
) -> MultiAttemptResult:
    """Try tactics using LSP file modifications (fallback)."""
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        raise LeanToolError(
            "Invalid Lean file path: Unable to start LSP server or load file"
        )

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    original_content = get_file_contents(file_path)

    try:
        results: List[AttemptResult] = []
        for snippet in snippets:
            snippet_str = snippet.rstrip("\n")
            payload = f"{snippet_str}\n"
            change = DocumentContentChange(
                payload,
                [line - 1, 0],
                [line, 0],
            )
            client.update_file(rel_path, [change])
            diag = client.get_diagnostics(rel_path)
            check_lsp_response(diag, "get_diagnostics")
            filtered_diag = filter_diagnostics_by_position(diag, line - 1, None)
            goal_result = client.get_goal(rel_path, line - 1, len(snippet_str))
            goals = extract_goals_list(goal_result)
            results.append(
                AttemptResult(
                    snippet=snippet_str,
                    goals=goals,
                    diagnostics=_to_diagnostic_messages(filtered_diag),
                )
            )

        return MultiAttemptResult(items=results)
    finally:
        if original_content is not None:
            try:
                client.update_file_content(rel_path, original_content)
            except Exception as exc:
                logger.warning(
                    "Failed to restore `%s` after multi_attempt: %s", rel_path, exc
                )


@mcp.tool(
    "lean_multi_attempt",
    annotations=ToolAnnotations(
        title="Multi-Attempt",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def multi_attempt(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    snippets: Annotated[
        List[str],
        Field(description="Tactics to try (3+ recommended)"),
    ],
) -> MultiAttemptResult:
    """Try multiple tactics without modifying file. Returns goal state for each."""
    # Priority 1: REPL
    result = await _multi_attempt_repl(ctx, file_path, line, snippets)
    if result is not None:
        return result

    # Priority 2: LSP approach (fallback)
    return _multi_attempt_lsp(ctx, file_path, line, snippets)


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
        check_lsp_response(raw_diagnostics, "get_diagnostics")
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
async def local_search(
    ctx: Context,
    query: Annotated[str, Field(description="Declaration name or prefix")],
    limit: Annotated[int, Field(description="Max matches", ge=1)] = 10,
    project_root: Annotated[
        Optional[str], Field(description="Project root (inferred if omitted)")
    ] = None,
) -> LocalSearchResults:
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
        raw_results = await asyncio.to_thread(
            lean_local_search,
            query=query.strip(),
            limit=limit,
            project_root=resolved_root,
        )
        results = [
            LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
            for r in raw_results
        ]
        return LocalSearchResults(items=results)
    except RuntimeError as exc:
        raise LocalSearchError(f"Search failed: {exc}")


def _leanexplore_base_url() -> str:
    base_url = os.environ.get("LEAN_EXPLORE_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://www.leanexplore.com/api/v1"
    return base_url.rstrip("/")


def _leanexplore_use_local_backend() -> bool:
    backend = os.environ.get(_LEANEXPLORE_BACKEND_ENV, "").strip().lower()
    if backend in ("local", "1", "true", "yes"):
        return True
    flag = os.environ.get(_LEANEXPLORE_LOCAL_ENV, "").strip().lower()
    return flag in ("1", "true", "yes")


def _leanexplore_default_rerank_top() -> int | None:
    raw = os.environ.get("LEAN_EXPLORE_RERANK_TOP", "").strip()
    if not raw:
        return 50
    lowered = raw.lower()
    if lowered in ("none", "off", "disabled", "disable", "null"):
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise LeanToolError(
            "Invalid LEAN_EXPLORE_RERANK_TOP value. "
            "Use an integer >= 0 or disable with `none`."
        ) from exc
    if value < 0:
        raise LeanToolError(
            "Invalid LEAN_EXPLORE_RERANK_TOP value. "
            "Use an integer >= 0 or disable with `none`."
        )
    return value


def _leanexplore_get_local_service(app_ctx: AppContext):
    if app_ctx.leanexplore_service is not None:
        return app_ctx.leanexplore_service

    LeanExploreLocalService = None
    import_errors: List[Exception] = []
    for module_path in ("lean_explore.search.service", "lean_explore.local.service"):
        try:
            module = __import__(module_path, fromlist=["Service"])
            LeanExploreLocalService = getattr(module, "Service")
            break
        except Exception as exc:
            import_errors.append(exc)

    if LeanExploreLocalService is None:
        root_cause = import_errors[-1] if import_errors else None
        raise LeanToolError(
            "LeanExplore local backend requested but lean-explore is not installed. "
            "Install with `uv add 'lean-explore[local]'` and fetch data via "
            "`lean-explore data fetch`."
        ) from root_cause

    try:
        app_ctx.leanexplore_service = LeanExploreLocalService()
    except Exception as exc:
        details = str(exc).strip()
        if details:
            details = f" ({details})"
        raise LeanToolError(
            "LeanExplore local backend failed to initialize"
            f"{details}. Ensure `lean-explore[local]` is installed and run "
            "`lean-explore data fetch`."
        ) from exc

    return app_ctx.leanexplore_service


def _leanexplore_item_to_dict(item: object) -> dict:
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if hasattr(item, "dict"):
        return item.dict()
    return {}


def _leanexplore_extract_items(payload: object, key: str) -> List[object]:
    if payload is None:
        return []
    if isinstance(payload, list):
        flattened: List[object] = []
        for entry in payload:
            flattened.extend(_leanexplore_extract_items(entry, key))
        if flattened:
            return flattened
        return payload
    if hasattr(payload, key):
        items = getattr(payload, key)
        return items if isinstance(items, list) else []
    if isinstance(payload, dict):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
        if "items" in payload and isinstance(payload["items"], list):
            return payload["items"]
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
    return []


def _leanexplore_results_from_payload(
    payload: object, key: str
) -> List["LeanExploreResult"]:
    raw_items = _leanexplore_extract_items(payload, key)
    if not raw_items and key != "results":
        raw_items = _leanexplore_extract_items(payload, "results")
    if not raw_items and key != "citations":
        raw_items = _leanexplore_extract_items(payload, "citations")
    return [_leanexplore_parse_item(item) for item in raw_items if item is not None]


def _leanexplore_headers() -> Dict[str, str]:
    headers = {"User-Agent": "lean-lsp-mcp/0.1"}
    api_key = os.environ.get("LEAN_EXPLORE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _leanexplore_request_json(
    url: str, *, allow_404: bool = False
) -> dict | None:
    req = urllib.request.Request(url, headers=_leanexplore_headers(), method="GET")
    try:
        payload = await _urlopen_json(req, timeout=10)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"results": payload}
        raise LeanToolError("LeanExplore returned unexpected response shape.")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise LeanToolError(
                "LeanExplore authentication failed. Set LEAN_EXPLORE_API_KEY or "
                "use a local LEAN_EXPLORE_BASE_URL."
            ) from exc
        if exc.code == 404:
            if allow_404:
                return None
            raise LeanToolError("LeanExplore resource not found.") from exc
        raise LeanToolError(f"LeanExplore API error: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LeanToolError(f"LeanExplore API request failed: {exc.reason}") from exc
    except (orjson.JSONDecodeError, ValueError) as exc:
        raise LeanToolError("LeanExplore returned invalid JSON.") from exc


async def _leanexplore_await_if_needed(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


async def _leanexplore_local_search(
    service: object,
    *,
    query: str,
    package_filters: List[str] | None,
    limit: int,
    rerank_top: int | None,
) -> object:
    search_fn = getattr(service, "search", None)
    if search_fn is None:
        raise LeanToolError("LeanExplore local backend does not expose `search`.")

    attempts: List[dict[str, object]] = []

    def add_attempt(kwargs: dict[str, object]) -> None:
        if kwargs not in attempts:
            attempts.append(kwargs)

    package_variants: List[dict[str, object]] = [{}]
    if package_filters is not None:
        package_variants = [
            {"package_filters": package_filters},
            {"packages": package_filters},
        ]

    for package_kwargs in package_variants:
        with_rerank = {"query": query, "limit": limit, **package_kwargs}
        if rerank_top is not None:
            with_rerank["rerank_top"] = rerank_top
        add_attempt(with_rerank)
        add_attempt({"query": query, "limit": limit, **package_kwargs})

    if rerank_top is not None:
        add_attempt({"query": query, "limit": limit, "rerank_top": rerank_top})
    add_attempt({"query": query, "limit": limit})

    last_exc: TypeError | None = None
    for kwargs in attempts:
        try:
            result = search_fn(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        return await _leanexplore_await_if_needed(result)

    raise LeanToolError(
        f"LeanExplore local search failed due to API mismatch. Last error: {last_exc}"
    )


async def _leanexplore_local_get_by_id(service: object, group_id: int) -> object | None:
    get_by_id_fn = getattr(service, "get_by_id", None)
    if get_by_id_fn is None:
        raise LeanToolError("LeanExplore local backend does not expose `get_by_id`.")

    attempts = [
        ("kwargs", {"group_id": group_id}),
        ("kwargs", {"declaration_id": group_id}),
        ("args", (group_id,)),
    ]
    last_exc: TypeError | None = None
    for mode, payload in attempts:
        try:
            if mode == "kwargs":
                result = get_by_id_fn(**payload)
            else:
                result = get_by_id_fn(*payload)
        except TypeError as exc:
            last_exc = exc
            continue
        resolved = await _leanexplore_await_if_needed(result)
        if isinstance(resolved, list):
            if not resolved:
                return None
            return resolved[0]
        return resolved

    raise LeanToolError(
        "LeanExplore local `get_by_id` call failed due to API mismatch. "
        f"Last error: {last_exc}"
    )


async def _leanexplore_local_get_dependencies(
    service: object, group_id: int
) -> object | None:
    get_dependencies_fn = getattr(service, "get_dependencies", None)
    if get_dependencies_fn is None:
        # lean-explore v1.x removed the dedicated dependencies endpoint.
        return await _leanexplore_local_get_by_id(service, group_id)

    attempts = [
        ("kwargs", {"group_id": group_id}),
        ("kwargs", {"declaration_id": group_id}),
        ("args", (group_id,)),
    ]
    last_exc: TypeError | None = None
    for mode, payload in attempts:
        try:
            if mode == "kwargs":
                result = get_dependencies_fn(**payload)
            else:
                result = get_dependencies_fn(*payload)
        except TypeError as exc:
            last_exc = exc
            continue
        resolved = await _leanexplore_await_if_needed(result)
        if isinstance(resolved, list):
            if not resolved:
                return None
            return resolved[0]
        return resolved

    raise LeanToolError(
        "LeanExplore local `get_dependencies` call failed due to API mismatch. "
        f"Last error: {last_exc}"
    )


def _leanexplore_extract_dependency_names(payload: object) -> List[str]:
    raw = _leanexplore_item_to_dict(payload)
    raw_dependencies = raw.get("dependencies")
    if raw_dependencies is None:
        return []

    parsed_dependencies: object = raw_dependencies
    if isinstance(raw_dependencies, str):
        text = raw_dependencies.strip()
        if not text:
            return []
        try:
            parsed_dependencies = json.loads(text)
        except json.JSONDecodeError:
            parsed_dependencies = [part.strip() for part in text.split(",")]

    if not isinstance(parsed_dependencies, list):
        return []

    names: List[str] = []
    for dep in parsed_dependencies:
        if isinstance(dep, str):
            dep_name = dep.strip()
            if dep_name:
                names.append(dep_name)
        elif isinstance(dep, dict):
            dep_name = dep.get("name") or dep.get("lean_name")
            if isinstance(dep_name, str) and dep_name.strip():
                names.append(dep_name.strip())
    return names


def _leanexplore_dependency_names_to_results(
    names: List[str],
) -> List[LeanExploreResult]:
    results: List[LeanExploreResult] = []
    for idx, name in enumerate(names, start=1):
        results.append(
            LeanExploreResult(
                id=-idx,
                lean_name=name,
                source_file="",
                range_start_line=0,
                statement_text=name,
            )
        )
    return results


def _leanexplore_parse_item(item: object) -> LeanExploreResult:
    raw = _leanexplore_item_to_dict(item)
    primary = raw.get("primary_declaration") or raw.get("primaryDeclaration") or {}
    lean_name = raw.get("lean_name") or raw.get("leanName") or raw.get("name")
    if isinstance(primary, dict):
        lean_name = lean_name or primary.get("lean_name") or primary.get("leanName")
    statement_text = (
        raw.get("statement_text")
        or raw.get("display_statement_text")
        or raw.get("statement")
        or raw.get("displayStatementText")
        or raw.get("source_text")
        or raw.get("sourceText")
        or ""
    )
    source_file = (
        raw.get("source_file")
        or raw.get("sourceFile")
        or raw.get("file")
        or raw.get("file_path")
        or raw.get("module")
        or ""
    )
    range_start_line = raw.get("range_start_line")
    if range_start_line is None:
        range_data = raw.get("range") or {}
        if isinstance(range_data, dict):
            start = range_data.get("start", {})
            if isinstance(start, dict):
                range_start_line = start.get("line")
    if range_start_line is None:
        source_link = raw.get("source_link") or raw.get("sourceLink")
        if isinstance(source_link, str):
            match = re.search(r"#L(\d+)", source_link)
            if match:
                range_start_line = int(match.group(1))

    identifier = (
        raw.get("id") or raw.get("statement_group_id") or raw.get("statementGroupId")
    )
    if identifier is None:
        identifier = 0
    return LeanExploreResult(
        id=int(identifier),
        lean_name=lean_name,
        source_file=source_file,
        range_start_line=int(range_start_line or 0),
        statement_text=statement_text,
        docstring=raw.get("docstring"),
        informal_description=raw.get("informal_description")
        or raw.get("informalDescription")
        or raw.get("informalization"),
    )


@mcp.tool(
    "lean_leanexplore_search",
    annotations=ToolAnnotations(
        title="LeanExplore Search",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leanexplore", max_requests=3, per_seconds=30)
async def leanexplore_search(
    ctx: Context,
    query: Annotated[str, Field(description="Search query")],
    package_filters: Annotated[
        Optional[List[str]],
        Field(description="Optional list of package filters"),
    ] = None,
    rerank_top: Annotated[
        Optional[int],
        Field(
            description=(
                "Local backend only: number of candidates to rerank. "
                "Defaults to 50; use 0 to disable reranking."
            ),
            ge=0,
        ),
    ] = None,
    limit: Annotated[int, Field(description="Max results", ge=1)] = 10,
) -> LeanExploreResults:
    """Search Lean declarations via LeanExplore (API or local backend)."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    resolved_rerank_top = (
        _leanexplore_default_rerank_top() if rerank_top is None else rerank_top
    )

    if app_ctx.leanexplore_local_enabled:
        service = _leanexplore_get_local_service(app_ctx)
        try:
            data = await _leanexplore_local_search(
                service,
                query=query.strip(),
                package_filters=package_filters,
                limit=limit,
                rerank_top=resolved_rerank_top,
            )
        except Exception as exc:
            raise LeanToolError(f"LeanExplore local search failed: {exc}") from exc
        items = _leanexplore_results_from_payload(data, "results")[:limit]
        return LeanExploreResults(items=items)

    params: List[tuple[str, str]] = [("q", query.strip()), ("limit", str(limit))]
    if package_filters:
        # Legacy API (<1.0): repeated pkg params.
        params.extend([("pkg", pkg) for pkg in package_filters])
        # Newer API (>=1.0): CSV packages param.
        params.append(("packages", ",".join(package_filters)))

    query_string = urllib.parse.urlencode(params, doseq=True)
    url = f"{_leanexplore_base_url()}/search?{query_string}"
    data = await _leanexplore_request_json(url)
    if data is None:
        return LeanExploreResults(items=[])
    items = _leanexplore_results_from_payload(data, "results")[:limit]
    return LeanExploreResults(items=items)


@mcp.tool(
    "lean_leanexplore_get_by_id",
    annotations=ToolAnnotations(
        title="LeanExplore Get by ID",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leanexplore", max_requests=3, per_seconds=30)
async def leanexplore_get_by_id(
    ctx: Context,
    group_id: Annotated[int, Field(description="Statement group ID")],
) -> LeanExploreResult:
    """Fetch a LeanExplore statement group by ID."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if app_ctx.leanexplore_local_enabled:
        service = _leanexplore_get_local_service(app_ctx)
        try:
            data = await _leanexplore_local_get_by_id(service, group_id)
        except Exception as exc:
            raise LeanToolError(f"LeanExplore local lookup failed: {exc}") from exc
        if data is None:
            raise LeanToolError("LeanExplore resource not found.")
        return _leanexplore_parse_item(data)

    base_url = _leanexplore_base_url()
    data = await _leanexplore_request_json(
        f"{base_url}/declarations/{group_id}", allow_404=True
    )
    if data is None:
        data = await _leanexplore_request_json(
            f"{base_url}/statement_groups/{group_id}", allow_404=True
        )
    if data is None:
        raise LeanToolError("LeanExplore resource not found.")
    return _leanexplore_parse_item(data)


@mcp.tool(
    "lean_leanexplore_dependencies",
    annotations=ToolAnnotations(
        title="LeanExplore Dependencies",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leanexplore", max_requests=3, per_seconds=30)
async def leanexplore_dependencies(
    ctx: Context,
    group_id: Annotated[int, Field(description="Statement group ID")],
    limit: Annotated[int, Field(description="Max results", ge=1)] = 10,
) -> LeanExploreResults:
    """Fetch dependency (citation) groups for a LeanExplore statement group."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if app_ctx.leanexplore_local_enabled:
        service = _leanexplore_get_local_service(app_ctx)
        try:
            data = await _leanexplore_local_get_dependencies(service, group_id)
        except Exception as exc:
            raise LeanToolError(
                f"LeanExplore local dependency lookup failed: {exc}"
            ) from exc
        if data is None:
            return LeanExploreResults(items=[])
        items = _leanexplore_results_from_payload(data, "citations")
        if not items:
            dep_names = _leanexplore_extract_dependency_names(data)
            items = _leanexplore_dependency_names_to_results(dep_names)
        items = items[:limit]
        return LeanExploreResults(items=items)

    base_url = _leanexplore_base_url()
    dependency_payload = None
    for url in (
        f"{base_url}/declarations/{group_id}/dependencies",
        f"{base_url}/statement_groups/{group_id}/dependencies",
    ):
        dependency_payload = await _leanexplore_request_json(url, allow_404=True)
        if dependency_payload is not None:
            break

    if dependency_payload is not None:
        items = _leanexplore_results_from_payload(dependency_payload, "citations")
        if items:
            return LeanExploreResults(items=items[:limit])

        dep_names = _leanexplore_extract_dependency_names(dependency_payload)
        if dep_names:
            return LeanExploreResults(
                items=_leanexplore_dependency_names_to_results(dep_names[:limit])
            )

    declaration_payload = await _leanexplore_request_json(
        f"{base_url}/declarations/{group_id}", allow_404=True
    )
    if declaration_payload is None:
        declaration_payload = await _leanexplore_request_json(
            f"{base_url}/statement_groups/{group_id}", allow_404=True
        )
    if declaration_payload is None:
        return LeanExploreResults(items=[])

    dependency_names = _leanexplore_extract_dependency_names(declaration_payload)
    return LeanExploreResults(
        items=_leanexplore_dependency_names_to_results(dependency_names[:limit])
    )


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
async def leansearch(
    ctx: Context,
    query: Annotated[str, Field(description="Natural language or Lean term query")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> LeanSearchResults:
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

    await _safe_report_progress(
        ctx, progress=1, total=10, message="Awaiting response from leansearch.net"
    )
    results = await _urlopen_json(req, timeout=10)

    if not results or not results[0]:
        return LeanSearchResults(items=[])

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
    return LeanSearchResults(items=items)


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
) -> LoogleResults:
    """Search Mathlib by type signature via loogle.lean-lang.org.

    Examples: `Real.sin`, `"comm"`, `(?a → ?b) → List ?a → List ?b`,
    `_ * (_ ^ _)`, `|- _ < _ → _ + 1 < _ + 1`
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Try local loogle first if available (no rate limiting)
    if app_ctx.loogle_local_available and app_ctx.loogle_manager:
        # Update project path if it changed (adds new library paths)
        if app_ctx.lean_project_path != app_ctx.loogle_manager.project_path:
            if app_ctx.loogle_manager.set_project_path(app_ctx.lean_project_path):
                # Restart to pick up new paths
                await app_ctx.loogle_manager.stop()
        try:
            results = await app_ctx.loogle_manager.query(query, num_results)
            if not results:
                return LoogleResults(items=[])
            items = [
                LoogleResult(
                    name=r.get("name", ""),
                    type=r.get("type", ""),
                    module=r.get("module", ""),
                )
                for r in results
            ]
            return LoogleResults(items=items)
        except Exception as e:
            logger.warning(f"Local loogle failed: {e}, falling back to remote")

    # Fall back to remote (with rate limiting)
    rate_limit = app_ctx.rate_limit["loogle"]
    now = int(time.time())
    rate_limit[:] = [t for t in rate_limit if now - t < 30]
    if len(rate_limit) >= 3:
        raise LeanToolError(
            "Rate limit exceeded: 3 requests per 30s. Use --loogle-local to avoid limits."
        )
    rate_limit.append(now)

    await _safe_report_progress(
        ctx,
        progress=1,
        total=10,
        message="Awaiting response from loogle.lean-lang.org",
    )
    result = await asyncio.to_thread(loogle_remote, query, num_results)
    if isinstance(result, str):
        raise LeanToolError(result)  # Error message from remote
    return LoogleResults(items=result)


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
async def leanfinder(
    ctx: Context,
    query: Annotated[str, Field(description="Mathematical concept or proof state")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> LeanFinderResults:
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
    await _safe_report_progress(
        ctx,
        progress=1,
        total=10,
        message="Awaiting response from Lean Finder (Hugging Face)",
    )
    data = await _urlopen_json(req, timeout=10)
    for result in data["results"]:
        if (
            "https://leanprover-community.github.io/mathlib4_docs" not in result["url"]
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

    return LeanFinderResults(items=results)


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
async def state_search(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> StateSearchResults:
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

    await _safe_report_progress(
        ctx, progress=1, total=10, message=f"Awaiting response from {url}"
    )
    results = await _urlopen_json(req, timeout=10)

    items = [StateSearchResult(name=r["name"]) for r in results]
    return StateSearchResults(items=items)


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
async def hammer_premise(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 32,
) -> PremiseResults:
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

    await _safe_report_progress(
        ctx, progress=1, total=10, message=f"Awaiting response from {url}"
    )
    results = await _urlopen_json(req, timeout=10)

    items = [PremiseResult(name=r["name"]) for r in results]
    return PremiseResults(items=items)


@mcp.tool(
    "lean_profile_proof",
    annotations=ToolAnnotations(
        title="Profile Proof",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profile_proof(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    line: Annotated[
        int, Field(description="Line where theorem starts (1-indexed)", ge=1)
    ],
    top_n: Annotated[
        int, Field(description="Number of slowest lines to return", ge=1)
    ] = 5,
    timeout: Annotated[float, Field(description="Max seconds to wait", ge=1)] = 60.0,
) -> ProofProfileResult:
    """Run `lean --profile` on a theorem. Returns per-line timing and categories."""
    from lean_lsp_mcp.profile_utils import profile_theorem

    # Get project path
    lifespan = ctx.request_context.lifespan_context
    project_path = lifespan.lean_project_path

    if not project_path:
        infer_project_path(ctx, file_path)
        project_path = lifespan.lean_project_path

    if not project_path:
        raise LeanToolError("Lean project not found")

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise LeanToolError(f"File not found: {file_path}")

    try:
        return await profile_theorem(
            file_path=file_path_obj,
            theorem_line=line,
            project_path=project_path,
            timeout=timeout,
            top_n=top_n,
        )
    except (ValueError, TimeoutError) as e:
        raise LeanToolError(str(e)) from e


if __name__ == "__main__":
    mcp.run()
