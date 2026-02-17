import argparse
import os

from lean_lsp_mcp.server import mcp


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
    parser.add_argument(
        "--hammer-local",
        action="store_true",
        help="Enable local lean-hammer premise server (requires Docker or macOS container). "
        "Avoids rate limits and network dependencies.",
    )
    parser.add_argument(
        "--hammer-port",
        type=int,
        default=8765,
        help="Port for local hammer server (default: 8765)",
    )
    parser.add_argument(
        "--hammer-local-only",
        action="store_true",
        help="Require local hammer and disable remote fallback.",
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
    if args.hammer_local:
        os.environ["LEAN_HAMMER_LOCAL"] = "true"
    if args.hammer_port != 8765:
        os.environ["LEAN_HAMMER_PORT"] = str(args.hammer_port)
    if args.hammer_local_only:
        os.environ["LEAN_HAMMER_LOCAL_ONLY"] = "true"

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        return 130
    return 0
