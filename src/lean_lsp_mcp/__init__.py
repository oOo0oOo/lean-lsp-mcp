import argparse
import os
import sys
from contextlib import suppress

import anyio
from lean_lsp_mcp.server import mcp

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
        "--loogle-local",
        action="store_true",
        help="Enable local loogle (auto-installs on first run, ~5-10 min). "
        "Avoids rate limits and network dependencies.",
    )
    parser.add_argument(
        "--loogle-cache-dir",
        type=str,
        help="Override loogle cache location (default: ~/.cache/lean-lsp-mcp/loogle)",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Enable fast REPL-based multi-attempt (~5x faster). Requires Lean REPL.",
    )
    parser.add_argument(
        "--repl-timeout",
        type=int,
        help="REPL command timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    # Set env vars from CLI args (CLI takes precedence over env vars)
    if args.loogle_local:
        os.environ["LEAN_LOOGLE_LOCAL"] = "true"
    if args.loogle_cache_dir:
        os.environ["LEAN_LOOGLE_CACHE_DIR"] = args.loogle_cache_dir
    if args.repl:
        os.environ["LEAN_REPL"] = "true"
    if args.repl_timeout:
        os.environ["LEAN_REPL_TIMEOUT"] = str(args.repl_timeout)
    os.environ["LEAN_LSP_MCP_ACTIVE_TRANSPORT"] = args.transport

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.transport == "stdio" and _is_transport_closed_error(exc):
            _silence_stdout()
            return 0
        raise
    return 0
