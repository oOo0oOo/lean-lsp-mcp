import asyncio
import os
import re
import time
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import urllib
import orjson
import functools
import subprocess
import uuid
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger, configure_logging
from mcp.server.auth.settings import AuthSettings
from leanclient import LeanLSPClient, DocumentContentChange

from lean_lsp_mcp.client_utils import (
    setup_client_for_file,
    startup_client,
    infer_project_path,
)
from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.utils import (
    OutputCapture,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_diagnostics,
    format_goal,
    format_line,
    get_declaration_range,
    OptionalTokenVerifier,
)


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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    try:
        lean_project_path_str = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        if not lean_project_path_str:
            lean_project_path = None
        else:
            lean_project_path = Path(lean_project_path_str).resolve()

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
        )
        yield context
    finally:
        logger.info("Closing Lean LSP client")

        if context.client:
            context.client.close()


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
@mcp.tool("lean_build")
async def lsp_build(
    ctx: Context, lean_project_path: str = None, clean: bool = False
) -> str:
    """Build the Lean project and restart the LSP Server.

    Use only if needed (e.g. new imports).

    Args:
        lean_project_path (str, optional): Path to the Lean project. If not provided, it will be inferred from previous tool calls.
        clean (bool, optional): Run `lake clean` before building. Attention: Only use if it is really necessary! It can take a long time! Defaults to False.

    Returns:
        str: Build output or error msg
    """
    if not lean_project_path:
        lean_project_path_obj = ctx.request_context.lifespan_context.lean_project_path
    else:
        lean_project_path_obj = Path(lean_project_path).resolve()
        ctx.request_context.lifespan_context.lean_project_path = lean_project_path_obj

    if lean_project_path_obj is None:
        return "Lean project path not known yet. Provide `lean_project_path` explicitly or call a tool that infers it (e.g. `lean_file_contents`) before running `lean_build`."

    build_output = ""
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

        output_lines = []

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_str = line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line_str)

            # Parse progress: look for pattern like "[2/8]" or "[10/100]"
            match = re.search(r"\[(\d+)/(\d+)\]", line_str)
            if match:
                current_job = int(match.group(1))
                total_jobs = int(match.group(2))

                # Extract what's being built
                # Line format: "ℹ [2/8] Built TestLeanBuild.Basic (1.6s)"
                desc_match = re.search(
                    r"\[\d+/\d+\]\s+(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str
                )
                description = desc_match.group(1) if desc_match else "Building"

                # Report progress using dynamic totals from Lake
                await ctx.report_progress(
                    progress=current_job, total=total_jobs, message=description
                )

        await process.wait()

        if process.returncode != 0:
            build_output = "\n".join(output_lines)
            raise Exception(f"Build failed with return code {process.returncode}")

        # Start LSP client (without initial build since we just did it)
        with OutputCapture():
            client = LeanLSPClient(
                lean_project_path_obj, initial_build=False, prevent_cache_get=True
            )

        logger.info("Built project and re-started LSP client")

        ctx.request_context.lifespan_context.client = client
        build_output = "\n".join(output_lines)
        return build_output
    except Exception as e:
        return f"Error during build:\n{str(e)}\n{build_output}"


# File level tools
@mcp.tool("lean_file_contents")
def file_contents(ctx: Context, file_path: str, annotate_lines: bool = True) -> str:
    """Get the text contents of a Lean file, optionally with line numbers.

    Use sparingly (bloats context). Mainly when unsure about line numbers.

    Args:
        file_path (str): Abs path to Lean file
        annotate_lines (bool, optional): Annotate lines with line numbers. Defaults to True.

    Returns:
        str: File content or error msg
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


@mcp.tool("lean_diagnostic_messages")
def diagnostic_messages(
    ctx: Context,
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    declaration_name: Optional[str] = None,
) -> List[str] | str:
    """Get all diagnostic msgs (errors, warnings, infos) for a Lean file.

    "no goals to be solved" means code may need removal.

    Args:
        file_path (str): Abs path to Lean file
        start_line (int, optional): Start line (1-indexed). Filters from this line.
        end_line (int, optional): End line (1-indexed). Filters to this line.
        declaration_name (str, optional): Name of a specific theorem/lemma/definition.
            If provided, only returns diagnostics within that declaration.
            Takes precedence over start_line/end_line.
            Slow, requires waiting for full file analysis.

    Returns:
        List[str] | str: Diagnostic msgs or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client

    # If declaration_name is provided, get its range and use that for filtering
    if declaration_name:
        decl_range = get_declaration_range(client, rel_path, declaration_name)
        if decl_range is None:
            return f"Declaration '{declaration_name}' not found in file. Check the name (case-sensitive) and try again."
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

    return format_diagnostics(diagnostics)


@mcp.tool("lean_goal")
def goal(ctx: Context, file_path: str, line: int, column: Optional[int] = None) -> str:
    """Get the proof goals (proof state) at a specific location in a Lean file.

    VERY USEFUL! Main tool to understand the proof state and its evolution!
    Returns "no goals" if solved.
    To see the goal at sorry, use the cursor before the "s".
    Avoid giving a column if unsure-default behavior works well.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => Both before and after the line.

    Returns:
        str: Goal(s) or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)

    if column is None:
        lines = content.splitlines()
        if line < 1 or line > len(lines):
            return "Line number out of range. Try elsewhere?"
        column_end = len(lines[line - 1])
        column_start = next(
            (i for i, c in enumerate(lines[line - 1]) if not c.isspace()), 0
        )
        goal_start = client.get_goal(rel_path, line - 1, column_start)
        goal_end = client.get_goal(rel_path, line - 1, column_end)

        if goal_start is None and goal_end is None:
            return f"No goals on line:\n{lines[line - 1]}\nTry another line?"

        start_text = format_goal(goal_start, "No goals at line start.")
        end_text = format_goal(goal_end, "No goals at line end.")
        return f"Goals on line:\n{lines[line - 1]}\nBefore:\n{start_text}\nAfter:\n{end_text}"

    else:
        goal = client.get_goal(rel_path, line - 1, column - 1)
        f_goal = format_goal(goal, "Not a valid goal position. Try elsewhere?")
        f_line = format_line(content, line, column)
        return f"Goals at:\n{f_line}\n{f_goal}"


@mcp.tool("lean_term_goal")
def term_goal(
    ctx: Context, file_path: str, line: int, column: Optional[int] = None
) -> str:
    """Get the expected type (term goal) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => end of line.

    Returns:
        str: Expected type or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    if column is None:
        lines = content.splitlines()
        if line < 1 or line > len(lines):
            return "Line number out of range. Try elsewhere?"
        column = len(content.splitlines()[line - 1])

    term_goal = client.get_term_goal(rel_path, line - 1, column - 1)
    f_line = format_line(content, line, column)
    if term_goal is None:
        return f"Not a valid term goal position:\n{f_line}\nTry elsewhere?"
    rendered = term_goal.get("goal", None)
    if rendered is not None:
        rendered = rendered.replace("```lean\n", "").replace("\n```", "")
    return f"Term goal at:\n{f_line}\n{rendered or 'No term goal found.'}"


@mcp.tool("lean_hover_info")
def hover(ctx: Context, file_path: str, line: int, column: int) -> str:
    """Get hover info (docs for syntax, variables, functions, etc.) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed). Make sure to use the start or within the term, not the end.

    Returns:
        str: Hover info or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    file_content = client.get_file_content(rel_path)
    hover_info = client.get_hover(rel_path, line - 1, column - 1)
    if hover_info is None:
        f_line = format_line(file_content, line, column)
        return f"No hover information at position:\n{f_line}\nTry elsewhere?"

    # Get the symbol and the hover information
    h_range = hover_info.get("range")
    symbol = extract_range(file_content, h_range)
    info = hover_info["contents"].get("value", "No hover information available.")
    info = info.replace("```lean\n", "").replace("\n```", "").strip()

    # Add diagnostics if available
    diagnostics = client.get_diagnostics(rel_path)
    filtered = filter_diagnostics_by_position(diagnostics, line - 1, column - 1)

    msg = f"Hover info `{symbol}`:\n{info}"
    if filtered:
        msg += "\n\nDiagnostics\n" + "\n".join(format_diagnostics(filtered))
    return msg


@mcp.tool("lean_completions")
def completions(
    ctx: Context, file_path: str, line: int, column: int, max_completions: int = 32
) -> str:
    """Get code completions at a location in a Lean file.

    Only use this on INCOMPLETE lines/statements to check available identifiers and imports:
    - Dot Completion: Displays relevant identifiers after a dot (e.g., `Nat.`, `x.`, or `Nat.ad`).
    - Identifier Completion: Suggests matching identifiers after part of a name.
    - Import Completion: Lists importable files after `import` at the beginning of a file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        max_completions (int, optional): Maximum number of completions to return. Defaults to 32

    Returns:
        str: List of possible completions or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    completions = client.get_completions(rel_path, line - 1, column - 1)
    formatted = [c["label"] for c in completions if "label" in c]
    f_line = format_line(content, line, column)

    if not formatted:
        return f"No completions at position:\n{f_line}\nTry elsewhere?"

    # Find the sort term: The last word/identifier before the cursor
    lines = content.splitlines()
    prefix = ""
    if 0 < line <= len(lines):
        text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
        if not text_before_cursor.endswith("."):
            prefix = re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[-1].lower()

    # Sort completions: prefix matches first, then contains, then alphabetical
    if prefix:

        def sort_key(item):
            item_lower = item.lower()
            if item_lower.startswith(prefix):
                return (0, item_lower)
            elif prefix in item_lower:
                return (1, item_lower)
            else:
                return (2, item_lower)

        formatted.sort(key=sort_key)
    else:
        formatted.sort(key=str.lower)

    # Truncate if too many results
    if len(formatted) > max_completions:
        remaining = len(formatted) - max_completions
        formatted = formatted[:max_completions] + [
            f"{remaining} more, keep typing to filter further"
        ]
    completions_text = "\n".join(formatted)
    return f"Completions at:\n{f_line}\n{completions_text}"


@mcp.tool("lean_declaration_file")
def declaration_file(ctx: Context, file_path: str, symbol: str) -> str:
    """Get the file contents where a symbol/lemma/class/structure is declared.

    Note:
        Symbol must be present in the file! Add if necessary!
        Lean files can be large, use `lean_hover_info` before this tool.

    Args:
        file_path (str): Abs path to Lean file
        symbol (str): Symbol to look up the declaration for. Case sensitive!

    Returns:
        str: File contents or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    orig_file_content = client.get_file_content(rel_path)

    # Find the first occurence of the symbol (line and column) in the file,
    position = find_start_position(orig_file_content, symbol)
    if not position:
        return f"Symbol `{symbol}` (case sensitive) not found in file `{rel_path}`. Add it first, then try again."

    declaration = client.get_declarations(
        rel_path, position["line"], position["column"]
    )

    if len(declaration) == 0:
        return f"No declaration available for `{symbol}`."

    # Load the declaration file
    declaration = declaration[0]
    uri = declaration.get("targetUri")
    if not uri:
        uri = declaration.get("uri")

    abs_path = client._uri_to_abs(uri)
    if not os.path.exists(abs_path):
        return f"Could not open declaration file `{abs_path}` for `{symbol}`."

    file_content = get_file_contents(abs_path)

    return f"Declaration of `{symbol}`:\n{file_content}"


@mcp.tool("lean_multi_attempt")
def multi_attempt(
    ctx: Context, file_path: str, line: int, snippets: List[str]
) -> List[str] | str:
    """Try multiple Lean code snippets at a line and get the goal state and diagnostics for each.

    Use to compare tactics or approaches.
    Use rarely-prefer direct file edits to keep users involved.
    For a single snippet, edit the file and run `lean_diagnostic_messages` instead.

    Note:
        Only single-line, fully-indented snippets are supported.
        Avoid comments for best results.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        snippets (List[str]): List of snippets (3+ are recommended)

    Returns:
        List[str] | str: Diagnostics and goal states or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

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
            formatted_diag = "\n".join(format_diagnostics(diag, select_line=line - 1))
            # Use the snippet text length without any trailing newline for the column
            goal = client.get_goal(rel_path, line - 1, len(snippet_str))
            formatted_goal = format_goal(goal, "Missing goal")
            results.append(f"{snippet_str}:\n {formatted_goal}\n\n{formatted_diag}")

        return results
    finally:
        try:
            client.close_files([rel_path])
        except Exception as exc:  # pragma: no cover - close failures only logged
            logger.warning(
                "Failed to close `%s` after multi_attempt: %s", rel_path, exc
            )


@mcp.tool("lean_run_code")
def run_code(ctx: Context, code: str) -> List[str] | str:
    """Run a complete, self-contained code snippet and return diagnostics.

    Has to include all imports and definitions!
    Only use for testing outside open files! Keep the user in the loop by editing files instead.

    Args:
        code (str): Code snippet

    Returns:
        List[str] | str: Diagnostics msgs or error msg
    """
    lifespan_context = ctx.request_context.lifespan_context
    lean_project_path = lifespan_context.lean_project_path
    if lean_project_path is None:
        return "No valid Lean project path found. Run another tool (e.g. `lean_file_contents`) first to set it up."

    # Use a unique snippet filename to avoid collisions under concurrency
    rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
    abs_path = lean_project_path / rel_path

    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        return f"Error writing code snippet to file `{abs_path}`:\n{str(e)}"

    client: LeanLSPClient | None = lifespan_context.client
    diagnostics: List[str] | str = []
    close_error: str | None = None
    remove_error: str | None = None
    opened_file = False

    try:
        if client is None:
            startup_client(ctx)
            client = lifespan_context.client
            if client is None:
                return "Failed to initialize Lean client for run_code."

        assert client is not None  # startup_client guarantees an initialized client
        client.open_file(rel_path)
        opened_file = True
        diagnostics = format_diagnostics(
            client.get_diagnostics(rel_path, inactivity_timeout=15.0)
        )
    finally:
        if opened_file:
            try:
                client.close_files([rel_path])
            except Exception as exc:  # pragma: no cover - close failures only logged
                close_error = str(exc)
                logger.warning("Failed to close `%s` after run_code: %s", rel_path, exc)
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            remove_error = str(e)
            logger.warning(
                "Failed to remove temporary Lean snippet `%s`: %s", abs_path, e
            )

    if remove_error:
        return f"Error removing temporary file `{abs_path}`:\n{remove_error}"
    if close_error:
        return f"Error closing temporary Lean document `{rel_path}`:\n{close_error}"

    return (
        diagnostics
        if diagnostics
        else "No diagnostics found for the code snippet (compiled successfully)."
    )


@mcp.tool("lean_local_search")
def local_search(
    ctx: Context, query: str, limit: int = 10
) -> List[Dict[str, str]] | str:
    """Confirm declarations exist in the current workspace to prevent hallucinating APIs.

    VERY USEFUL AND FAST!
    Pass a short prefix (e.g. ``map_mul``); the metadata shows the declaration kind and file.
    The index spans theorems, lemmas, defs, classes, instances, structures, inductives, abbrevs, and opaque decls.

    Args:
        query (str): Declaration name or prefix.
        limit (int): Max matches to return (default 10).

    Returns:
        List[Dict[str, str]] | str: Matches as ``{"name", "kind", "file"}`` or error message.
    """
    if not _RG_AVAILABLE:
        return _RG_MESSAGE

    stored_root = ctx.request_context.lifespan_context.lean_project_path
    if stored_root is None:
        return "Lean project path not set. Call a file-based tool (like lean_file_contents) first to set the project path."

    return lean_local_search(query=query.strip(), limit=limit, project_root=stored_root)


@mcp.tool("lean_leansearch")
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(ctx: Context, query: str, num_results: int = 5) -> List[Dict] | str:
    """Search for Lean theorems, definitions, and tactics using leansearch.net.

    Query patterns:
      - Natural language: "If there exist injective maps of sets from A to B and from B to A, then there exists a bijective map between A and B."
      - Mixed natural/Lean: "natural numbers. from: n < m, to: n + 1 < m + 1", "n + 1 <= m if n < m"
      - Concept names: "Cauchy Schwarz"
      - Lean identifiers: "List.sum", "Finset induction"
      - Lean term: "{f : A → B} {g : B → A} (hf : Injective f) (hg : Injective g) : ∃ h, Bijective h"

    Args:
        query (str): Search query
        num_results (int, optional): Max results. Defaults to 5.

    Returns:
        List[Dict] | str: Search results or error msg
    """
    try:
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
            return "No results found."
        results = results[0][:num_results]
        results = [r["result"] for r in results]

        for result in results:
            result.pop("docstring")
            result["module_name"] = ".".join(result["module_name"])
            result["name"] = ".".join(result["name"])

        return results
    except Exception as e:
        return f"leansearch error:\n{str(e)}"


@mcp.tool("lean_loogle")
@rate_limited("loogle", max_requests=3, per_seconds=30)
def loogle(ctx: Context, query: str, num_results: int = 8) -> List[dict] | str:
    """Search for definitions and theorems using loogle.

    Query patterns:
      - By constant: Real.sin  # finds lemmas mentioning Real.sin
      - By lemma name: "differ"  # finds lemmas with "differ" in the name
      - By subexpression: _ * (_ ^ _)  # finds lemmas with a product and power
      - Non-linear: Real.sqrt ?a * Real.sqrt ?a
      - By type shape: (?a -> ?b) -> List ?a -> List ?b
      - By conclusion: |- tsum _ = _ * tsum _
      - By conclusion w/hyps: |- _ < _ → tsum _ < tsum _

    Args:
        query (str): Search query
        num_results (int, optional): Max results. Defaults to 8.

    Returns:
        List[dict] | str: Search results or error msg
    """
    try:
        req = urllib.request.Request(
            f"https://loogle.lean-lang.org/json?q={urllib.parse.quote(query)}",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = orjson.loads(response.read())

        if "hits" not in results:
            return "No results found."

        results = results["hits"][:num_results]
        for result in results:
            result.pop("doc", None)
        return results
    except Exception as e:
        return f"loogle error:\n{str(e)}"


@mcp.tool("lean_leanfinder")
@rate_limited("leanfinder", max_requests=10, per_seconds=30)
def leanfinder(ctx: Context, query: str, num_results: int = 5) -> List[Dict] | str:
    """Search Mathlib theorems/definitions semantically by mathematical concept or proof state using Lean Finder.

    Effective query types:
    - Natural language mathematical statement: "For any natural numbers n and m, the sum n+m is equal to m+n."
    - Natural language questions: "I'm working with algebraic elements over a field extension … Does this imply that the minimal polynomials of x and y are equal?"
    - Proof state. For better results, enter a proof state followed by how you want to transform the proof state.
    - Statement definition: Fragment or the whole statement definition.

    Tips: Multiple targeted queries beat one complex query.

    Args:
        query (str): Mathematical concept or proof state
        num_results (int, optional): Max results. Defaults to 5.

    Returns:
        List[Dict] | str: List of Lean statement objects (full name, formal statement, informal statement) or error msg
    """
    try:
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
                ):  # Do not include results from other sources other than mathlib4, since users might not have imported them
                    continue
                full_name = re.search(r"pattern=(.*?)#doc", result["url"]).group(1)
                obj = {
                    "full_name": full_name,
                    "formal_statement": result["formal_statement"],
                    "informal_statement": result["informal_statement"],
                }
                results.append(obj)

        return results if results else "Lean Finder: No results parsed"
    except Exception as e:
        return f"Lean Finder Error:\n{str(e)}"


@mcp.tool("lean_state_search")
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 5
) -> List | str:
    """Search for theorems based on proof state using premise-search.com.

    Only uses first goal if multiple.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        num_results (int, optional): Max results. Defaults to 5.

    Returns:
        List | str: Search results or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    file_contents = client.get_file_content(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    f_line = format_line(file_contents, line, column)
    if not goal or not goal.get("goals"):
        return f"No goals found:\n{f_line}\nTry elsewhere?"

    goal = urllib.parse.quote(goal["goals"][0])

    try:
        url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
        req = urllib.request.Request(
            f"{url}/api/search?query={goal}&results={num_results}&rev=v4.22.0",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = orjson.loads(response.read())

        for result in results:
            result.pop("rev")
        # Very dirty type mix
        results.insert(0, f"Results for line:\n{f_line}")
        return results
    except Exception as e:
        return f"lean state search error:\n{str(e)}"


@mcp.tool("lean_hammer_premise")
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 32
) -> List[str] | str:
    """Search for premises based on proof state using the lean hammer premise search.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        num_results (int, optional): Max results. Defaults to 32.

    Returns:
        List[str] | str: List of relevant premises or error message
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    file_contents = client.get_file_content(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    f_line = format_line(file_contents, line, column)
    if not goal or not goal.get("goals"):
        return f"No goals found:\n{f_line}\nTry elsewhere?"

    data = {
        "state": goal["goals"][0],
        "new_premises": [],
        "k": num_results,
    }

    try:
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

        results = [result["name"] for result in results]
        results.insert(0, f"Results for line:\n{f_line}")
        return results
    except Exception as e:
        return f"lean hammer premise error:\n{str(e)}"


if __name__ == "__main__":
    mcp.run()
