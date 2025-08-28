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
    args = parser.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)
