import argparse
import asyncio
import os
import signal
import sys
import threading
from contextlib import suppress

import anyio
from lean_lsp_mcp import config
from lean_lsp_mcp.client_utils import close_shared_client, infer_project_path
from lean_lsp_mcp.server import VERSION, apply_tool_configuration, mcp

_TRANSPORT_CLOSE_HINTS = (
    "transport closed",
    "connection closed",
    "broken pipe",
)

_TRANSPORT_CLOSE_EXCEPTIONS = (
    BrokenPipeError,
    ConnectionResetError,
    EOFError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)


def _is_transport_closed_error(exc: BaseException) -> bool:
    """Walk exception tree (groups, __cause__, __context__) for transport errors."""
    stack: list[BaseException] = [exc]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, _TRANSPORT_CLOSE_EXCEPTIONS):
            return True
        if any(h in str(current).lower() for h in _TRANSPORT_CLOSE_HINTS):
            return True

        # Traverse ExceptionGroup-like .exceptions and chained causes
        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (tuple, list)):
            stack.extend(e for e in nested if isinstance(e, BaseException))
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                stack.append(linked)

    return False


def _silence_stdout() -> None:
    with suppress(Exception):
        sys.stdout.flush()

    with suppress(Exception):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Lean LSP MCP Server")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport method for the server. Default is 'stdio'.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port for transport",
    )
    parser.add_argument(
        "--lean-project-path",
        type=str,
        help=("Path to a Lean project root or to a file/dir inside it."),
    )
    parser.add_argument(
        "--disable-tools",
        type=str,
        help="Comma-separated tool names to disable (e.g. lean_run_code,lean_build).",
    )
    parser.add_argument(
        "--tool-descriptions",
        type=str,
        help="JSON object mapping tool names to replacement descriptions.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Override the server instructions sent to the client.",
    )
    parser.add_argument(
        "--loogle-local",
        action="store_true",
        help="Enable local loogle (auto-installs; first index may take a few "
        "minutes and substantial memory). Avoids rate limits.",
    )
    parser.add_argument(
        "--loogle-cache-dir",
        type=str,
        help="Override loogle cache location (default: ~/.cache/lean-lsp-mcp/loogle)",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Enable fast REPL-based run-code and multi-attempt. Requires Lean REPL.",
    )
    parser.add_argument(
        "--repl-timeout",
        type=int,
        help="REPL command timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    # Set env vars from CLI args (CLI takes precedence over env vars)
    if args.lean_project_path:
        project_path = infer_project_path(args.lean_project_path)
        if project_path is None:
            parser.error(f"No lean-toolchain found for: {args.lean_project_path}")
        os.environ[config.PROJECT_PATH_ENV] = str(project_path)
    if args.disable_tools:
        os.environ[config.DISABLED_TOOLS_ENV] = args.disable_tools
    if args.tool_descriptions:
        os.environ[config.TOOL_DESCRIPTIONS_ENV] = args.tool_descriptions
    if args.instructions:
        os.environ[config.INSTRUCTIONS_ENV] = args.instructions
    if args.loogle_local:
        os.environ[config.LOOGLE_LOCAL_ENV] = "true"
    if args.loogle_cache_dir:
        os.environ[config.LOOGLE_CACHE_DIR_ENV] = args.loogle_cache_dir
    if args.repl:
        os.environ[config.REPL_ENV] = "true"
    if args.repl_timeout:
        os.environ[config.REPL_TIMEOUT_ENV] = str(args.repl_timeout)
    os.environ[config.ACTIVE_TRANSPORT_ENV] = args.transport

    apply_tool_configuration(mcp)
    previous_sigterm = None
    sigterm_received = False
    if (
        args.transport == "stdio"
        and os.name == "posix"
        and threading.current_thread() is threading.main_thread()
    ):

        def cancel_on_sigterm(_signum, frame):
            nonlocal sigterm_received
            sigterm_received = True
            # AnyIO's SIGINT handler cancels startup before draining workers.
            # Background launches may inherit an ignored SIGINT instead.
            if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
                signal.default_int_handler(signal.SIGINT, frame)
            else:
                signal.raise_signal(signal.SIGINT)

        previous_sigterm = signal.signal(signal.SIGTERM, cancel_on_sigterm)
    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport=args.transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        if not sigterm_received:
            return 130
    except asyncio.CancelledError:
        if not sigterm_received:
            raise
    except Exception as exc:
        if args.transport == "stdio" and _is_transport_closed_error(exc):
            _silence_stdout()
        else:
            raise
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        close_shared_client()
    return 143 if sigterm_received else 0
