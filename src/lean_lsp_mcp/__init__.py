import argparse

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
    args = parser.parse_args()
    mcp.run(transport=args.transport)
