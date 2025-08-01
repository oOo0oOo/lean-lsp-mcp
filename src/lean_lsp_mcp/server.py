import os
import re
import time
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import urllib
import json
import functools
import subprocess

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from mcp.server.auth.settings import AuthSettings
from leanclient import LeanLSPClient, DocumentContentChange

from lean_lsp_mcp.client_utils import setup_client_for_file
from lean_lsp_mcp.file_utils import get_file_contents, update_file
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.utils import (
    OutputCapture,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_diagnostics,
    format_goal,
    format_line,
    OptionalTokenVerifier,
)


logger = get_logger(__name__)


# Server and context
@dataclass
class AppContext:
    lean_project_path: str | None
    client: LeanLSPClient | None
    file_content_hashes: Dict[str, str]
    rate_limit: Dict[str, List[int]]


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    try:
        lean_project_path = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        if not lean_project_path:
            lean_project_path = None
        else:
            lean_project_path = os.path.abspath(lean_project_path)

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            file_content_hashes={},
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
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
    lifespan=app_lifespan
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
            rate_limit = kwargs["ctx"].request_context.lifespan_context.rate_limit
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
def lsp_build(ctx: Context, lean_project_path: str = None, clean: bool = False) -> str:
    """Build the Lean project and restart the LSP Server.

    Use only if needed (e.g. new imports).

    Args:
        lean_project_path (str, optional): Path to the Lean project. If not provided, it will be inferred from previous tool calls.
        clean (bool, optional): Run `lake clean` before building. Attention: Only use if it is really necessary! It can take a long time! Defaults to False.

    Returns:
        str: Build output or error msg
    """
    if not lean_project_path:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    else:
        lean_project_path = os.path.abspath(lean_project_path)
        ctx.request_context.lifespan_context.lean_project_path = lean_project_path

    build_output = ""
    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        if client:
            client.close()
            ctx.request_context.lifespan_context.file_content_hashes.clear()

        if clean:
            subprocess.run(["lake", "clean"], cwd=lean_project_path, check=False)
            logger.info("Ran `lake clean`")

        with OutputCapture() as output:
            client = LeanLSPClient(
                lean_project_path,
                initial_build=True,
                print_warnings=False,
            )
        logger.info("Built project and re-started LSP client")

        ctx.request_context.lifespan_context.client = client
        build_output = output.get_output()
        return build_output
    except Exception as e:
        return f"Error during build:\n{str(e)}\n{build_output}"


# File level tools
@mcp.tool("lean_file_contents")
def file_contents(ctx: Context, file_path: str, annotate_lines: bool = True) -> str:
    """Get the text contents of a Lean file, optionally with line numbers.

    Args:
        file_path (str): Abs path to Lean file
        annotate_lines (bool, optional): Annotate lines with line numbers. Defaults to True.

    Returns:
        str: File content or error msg
    """
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
def diagnostic_messages(ctx: Context, file_path: str) -> List[str] | str:
    """Get all diagnostic msgs (errors, warnings, infos) for a Lean file.

    "no goals to be solved" means code may need removal.

    Args:
        file_path (str): Abs path to Lean file

    Returns:
        List[str] | str: Diagnostic msgs or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        return "Invalid Lean file path: Unable to start LSP server or load file"

    update_file(ctx, rel_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    diagnostics = client.get_diagnostics(rel_path)
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

    content = update_file(ctx, rel_path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client

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
        f_goal = format_goal(goal, f"Not a valid goal position. Try elsewhere?")
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

    content = update_file(ctx, rel_path)
    if column is None:
        lines = content.splitlines()
        if line < 1 or line > len(lines):
            return "Line number out of range. Try elsewhere?"
        column = len(content.splitlines()[line - 1])

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
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

    file_content = update_file(ctx, rel_path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
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
    content = update_file(ctx, rel_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
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
    orig_file_content = update_file(ctx, rel_path)

    # Find the first occurence of the symbol (line and column) in the file,
    position = find_start_position(orig_file_content, symbol)
    if not position:
        return f"Symbol `{symbol}` (case sensitive) not found in file `{rel_path}`. Add it first, then try again."

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
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
    update_file(ctx, rel_path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client

    client.open_file(rel_path)

    results = []
    snippets[0] += "\n"  # Extra newline for the first snippet
    for snippet in snippets:
        # Create a DocumentContentChange for the snippet
        change = DocumentContentChange(
            snippet + "\n",
            [line - 1, 0],
            [line, 0],
        )
        # Apply the change to the file, capture diagnostics and goal state
        diag = client.update_file(rel_path, [change])
        formatted_diag = "\n".join(format_diagnostics(diag, select_line=line - 1))
        goal = client.get_goal(rel_path, line - 1, len(snippet))
        formatted_goal = format_goal(goal, "Missing goal")
        results.append(f"{snippet}:\n {formatted_goal}\n\n{formatted_diag}")

    # Make sure it's clean after the attempts
    client.close_files([rel_path])
    return results


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
    lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    if lean_project_path is None:
        return "No valid Lean project path found. Run another tool (e.g. `lean_diagnostic_messages`) first to set it up or set the LEAN_PROJECT_PATH environment variable."

    rel_path = "temp_snippet.lean"
    abs_path = os.path.join(lean_project_path, rel_path)

    try:
        with open(abs_path, "w") as f:
            f.write(code)
    except Exception as e:
        return f"Error writing code snippet to file `{abs_path}`:\n{str(e)}"

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    diagnostics = format_diagnostics(client.get_diagnostics(rel_path))
    client.close_files([rel_path])

    try:
        os.remove(abs_path)
    except Exception as e:
        return f"Error removing temporary file `{abs_path}`:\n{str(e)}"

    return (
        diagnostics
        if diagnostics
        else "No diagnostics found for the code snippet (compiled successfully)."
    )


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
        payload = json.dumps(
            {"num_results": str(num_results), "query": [query]}
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://leansearch.net/search",
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

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
            results = json.loads(response.read().decode("utf-8"))

        if "hits" not in results:
            return "No results found."

        results = results["hits"][:num_results]
        for result in results:
            result.pop("doc")
        return results
    except Exception as e:
        return f"loogle error:\n{str(e)}"


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

    file_contents = update_file(ctx, rel_path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    goal = client.get_goal(rel_path, line - 1, column - 1)

    f_line = format_line(file_contents, line, column)
    if not goal or not goal.get("goals"):
        return f"No goals found:\n{f_line}\nTry elsewhere?"

    goal = urllib.parse.quote(goal["goals"][0])

    try:
        url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
        req = urllib.request.Request(
            f"{url}/api/search?query={goal}&results={num_results}&rev=v4.17.0-rc1",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

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

    file_contents = update_file(ctx, rel_path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
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
            data=json.dumps(data).encode("utf-8"),
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        results = [result["name"] for result in results]
        results.insert(0, f"Results for line:\n{f_line}")
        return results
    except Exception as e:
        return f"lean hammer premise error:\n{str(e)}"


if __name__ == "__main__":
    mcp.run()
