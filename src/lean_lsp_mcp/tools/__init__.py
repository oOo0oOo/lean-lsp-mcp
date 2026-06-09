"""MCP tool implementations.

Each submodule defines a thematic group of ``@mcp.tool`` handlers. Importing a
submodule registers its tools on the shared ``mcp`` instance (defined in
``lean_lsp_mcp.server``). Submodules reference shared infrastructure via the
``server`` module object so that test monkeypatching keeps working.
"""

from lean_lsp_mcp.tools import build, goals, widgets  # noqa: F401

__all__ = ["build", "goals", "widgets"]
