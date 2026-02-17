import argparse
import os

from lean_lsp_mcp.server import apply_tool_configuration, mcp


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
        "--project-root",
        type=str,
        help="Set LEAN_PROJECT_PATH explicitly (used for relative file paths).",
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
    if args.project_root:
        os.environ["LEAN_PROJECT_PATH"] = args.project_root
    if args.disable_tools:
        os.environ["LEAN_MCP_DISABLED_TOOLS"] = args.disable_tools
    if args.tool_descriptions:
        os.environ["LEAN_MCP_TOOL_DESCRIPTIONS"] = args.tool_descriptions
    if args.loogle_local:
        os.environ["LEAN_LOOGLE_LOCAL"] = "true"
    if args.loogle_cache_dir:
        os.environ["LEAN_LOOGLE_CACHE_DIR"] = args.loogle_cache_dir
    if args.repl:
        os.environ["LEAN_REPL"] = "true"
    if args.repl_timeout:
        os.environ["LEAN_REPL_TIMEOUT"] = str(args.repl_timeout)

    apply_tool_configuration(mcp)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)
