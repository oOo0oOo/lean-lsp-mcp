"""MCP tool implementations.

Each submodule defines a thematic group of ``@mcp.tool`` handlers. Importing a
submodule registers its tools on the shared ``mcp`` instance (defined in
``lean_lsp_mcp.server``). Submodules reference shared infrastructure via the
``server`` module object so that test monkeypatching keeps working.

Registration is driven by ``server`` (see ``TOOL_MODULES`` there), which imports
these submodules after the core is defined.
"""
