import os
import sys
import logging
from typing import List, Optional, Dict

from leanclient import LeanLSPClient

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from lean_lsp_mcp.prompts import PROMPT_AUTOMATIC_PROOF

# Configure logging to stderr instead of stdout to avoid interfering with LSP JSON communication
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger("lean-lsp-mcp")


class StdoutToStderr:
    """Redirects stdout to stderr at the file descriptor level bc lake build logging"""

    def __init__(self):
        self.original_stdout_fd = None

    def __enter__(self):
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = sys.stderr.fileno()
        os.dup2(stderr_fd, sys.stdout.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_stdout_fd is not None:
            os.dup2(self.original_stdout_fd, sys.stdout.fileno())
            os.close(self.original_stdout_fd)
            self.original_stdout_fd = None


# Lean project path management
LEAN_PROJECT_PATH = os.environ.get("LEAN_PROJECT_PATH", "").strip()
cwd = os.getcwd().strip()  # Strip necessary?
if not LEAN_PROJECT_PATH:
    logger.error("Please set the LEAN_PROJECT_PATH environment variable")
    sys.exit(1)


# File operations
def get_relative_file_path(file_path: str) -> Optional[str]:
    """Convert path relative to project path.

    Args:
        file_path (str): File path.

    Returns:
        str: Relative file path.
    """
    # Check if absolute path
    if os.path.exists(file_path):
        return os.path.relpath(file_path, LEAN_PROJECT_PATH)

    # Check if relative to project path
    path = os.path.join(LEAN_PROJECT_PATH, file_path)
    if os.path.exists(path):
        return os.path.relpath(path, LEAN_PROJECT_PATH)

    # Check if relative to CWD
    path = os.path.join(cwd, file_path)
    if os.path.exists(path):
        return os.path.relpath(path, LEAN_PROJECT_PATH)

    return None


def get_file_contents(rel_path: str) -> Optional[str]:
    with open(os.path.join(LEAN_PROJECT_PATH, rel_path), "r") as f:
        data = f.read()
    return data


def update_file(ctx: Context, rel_path: str) -> str:
    """Update the file contents in the context.
    Args:
        ctx (Context): Context object.
        rel_path (str): Relative file path.

    Returns:
        str: Updated file contents.
    """
    # Get file contents
    data = get_file_contents(rel_path)

    # Check if file_contents have changed
    file_contents: Dict[str, str] = ctx.request_context.lifespan_context.file_contents
    if rel_path not in file_contents:
        file_contents[rel_path] = data
        return data

    elif data == file_contents[rel_path]:
        return data

    # Update file_contents
    file_contents[rel_path] = data

    # Reload file in LSP
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.close_files([rel_path])
    return data


# Server and context
@dataclass
class AppContext:
    client: LeanLSPClient
    file_contents: Dict[str, str]


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    with StdoutToStderr():
        try:
            client = LeanLSPClient(LEAN_PROJECT_PATH)
            logger.info(f"Connected to Lean project at {LEAN_PROJECT_PATH}")
        except Exception as e:
            client = LeanLSPClient(LEAN_PROJECT_PATH, initial_build=False)
            logger.error(f"Could not do initial build, error: {e}")

    try:
        context = AppContext(client=client, file_contents={})
        yield context
    finally:
        logger.info("Closing Lean LSP client")
        context.client.close()


mcp = FastMCP(
    "Lean LSP",
    description="Interact with the Lean prover via the LSP",
    dependencies=["leanclient"],
    lifespan=app_lifespan,
    env_vars={
        "LEAN_PROJECT_PATH": {
            "description": "Path to the Lean project root",
            "required": True,
        }
    },
)


# Meta level tools
@mcp.tool("lean_auto_proof_instructions")
def auto_proof() -> str:
    """Get the description of the Lean LSP MCP and how to use it to automatically prove theorems.

    VERY IMPORTANT! Call this at the start of every proof and whenever you are unsure about the proof process.

    Returns:
        str: Description of the Lean LSP MCP.
    """
    return PROMPT_AUTOMATIC_PROOF


# Project level tools
@mcp.tool("lean_project_path")
def project_path() -> str:
    """Get the path to the Lean project root.

    Returns:
        str: Path to the Lean project.
    """
    return os.environ["LEAN_PROJECT_PATH"]


@mcp.tool("lean_project_functional")
def project_functional(ctx: Context) -> bool:
    """Check if the Lean project and the LSP are functional.

    Returns:
        bool: True if the Lean project is functional, False otherwise.
    """
    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.get_env(return_dict=False)
        return True
    except Exception:
        return False


@mcp.tool("lean_lsp_restart")
def lsp_restart(ctx: Context, rebuild: bool = True) -> bool:
    """Restart the LSP server. Can also rebuild the lean project.

    SLOW! Use only when necessary (e.g. imports) and in emergencies.

    Args:
        rebuild (bool, optional): Rebuild the Lean project. Defaults to True.

    Returns:
        bool: True if the Lean LSP server was restarted, False otherwise.
    """
    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.close()
        ctx.request_context.lifespan_context.client = LeanLSPClient(
            os.environ["LEAN_PROJECT_PATH"], initial_build=rebuild
        )
    except Exception:
        return False
    return True


# File level tools
@mcp.tool("lean_file_contents")
def file_contents(ctx: Context, file_path: str, annotate_lines: bool = True) -> str:
    """Get the text contents of a Lean file.

    IMPORTANT! Look up the file_contents for the currently open file including line number annotations.
    Use this during the proof process to keep updated on the line numbers and the current state of the file.

    Args:
        file_path (str): Absolute path to the Lean file.
        annotate_lines (bool, optional): Annotate lines with line numbers. Defaults to False.

    Returns:
        Optional[str]: Text contents of the Lean file or None if file does not exist.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    data = get_file_contents(rel_path)

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
    """Get all diagnostic messages for a Lean file.

    Attention! "no goals to be solved" indicates a mistake. Keep going!

    Args:
        file_path (str): Absolute path to the Lean file.

    Returns:
        List[str] | str: Diagnostic messages or error message.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    update_file(ctx, rel_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    diagnostics = client.get_diagnostics(rel_path)
    msgs = []
    # Format more compact
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range", None))
        if r is None:
            r_text = "No range"
        else:
            r_text = f"l{r['start']['line'] + 1}c{r['start']['character'] + 1} - l{r['end']['line'] + 1}c{r['end']['character'] + 1}"
        msgs.append(f"{r_text}, severity: {diag['severity']}\n{diag['message']}")
    return msgs


@mcp.tool("lean_goal")
def goal(ctx: Context, file_path: str, line: int, column: Optional[int] = None) -> str:
    """Get the proof goal at a specific location in a Lean file.

    VERY USEFUL AND CHEAP! This is your main tool to understand the proof state and its evolution!!
    Use this multiple times after every edit to the file!

    Solved proof state returns "no goals".

    Args:
        file_path (str): Absolute path to the Lean file.
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => end of line.

    Returns:
        str: Goal at the specified location or error message.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    content = update_file(ctx, rel_path)

    if column is None:
        column = len(content.splitlines()[line - 1])

    client: LeanLSPClient = ctx.request_context.lifespan_context.client

    goal = client.get_goal(rel_path, line - 1, column - 1)
    if goal is None:
        return "Not a valid goal position. Try again?"

    rendered = goal.get("rendered", None)
    if rendered is not None:
        rendered = rendered.replace("```lean\n", "").replace("\n```", "")
    return rendered


@mcp.tool("lean_term_goal")
def term_goal(
    ctx: Context, file_path: str, line: int, column: Optional[int] = None
) -> str:
    """Get the term goal at a specific location in a Lean file.

    Use this to get a better understanding of the proof state.

    Args:
        file_path (str): Absolute path to the Lean file.
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => end of line.

    Returns:
        str: Term goal at the specified location or error message.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    content = update_file(ctx, rel_path)
    if column is None:
        column = len(content.splitlines()[line - 1])

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    term_goal = client.get_term_goal(rel_path, line - 1, column - 1)
    if term_goal is None:
        return "Not a valid term goal position. Try again?"
    rendered = term_goal.get("goal", None)
    if rendered is not None:
        rendered = rendered.replace("```lean\n", "").replace("\n```", "")
    return rendered


@mcp.tool("lean_hover_info")
def hover(ctx: Context, file_path: str, line: int, column: int) -> str:
    """Get the hover information at a specific location in a Lean file.

    Use this information to look up information about lean syntax, variables, functions, etc.

    Args:
        file_path (str): Absolute path to the Lean file.
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed).

    Returns:
        str: Hover information at the specified location or error message.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    update_file(ctx, rel_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    hover_info = client.get_hover(rel_path, line - 1, column - 1)
    if hover_info is None:
        return "No hover information available. Try another position?"

    info = hover_info.get("contents", None)
    if info is not None:
        return info.get("value", "No hover information available.")


@mcp.tool("lean_proofs_complete")
def proofs_complete(ctx: Context, file_path: str) -> str:
    """Always check if all proofs in the file are complete in the end.

    Attention! "no goals to be solved" indicates a mistake. Keep going!

    Args:
        file_path (str): Absolute path to the Lean file.

    Returns:
        str: Message indicating if the proofs are complete or not.
    """
    rel_path = get_relative_file_path(file_path)
    if not rel_path:
        return "No valid lean file found."

    update_file(ctx, rel_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    diagnostics = client.get_diagnostics(rel_path)

    if diagnostics is None or len(diagnostics) > 0:
        return "Proof not complete! " + str(diagnostics)

    return "All proofs are complete!"


if __name__ == "__main__":
    mcp.run()
