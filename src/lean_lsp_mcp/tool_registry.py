"""Declarative MCP tool definitions with explicit server registration."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    options: dict[str, Any]


_DEFINITION_ATTR = "__lean_lsp_mcp_tool__"


def tool(name: str, **options: Any):
    """Mark a function as an MCP tool without binding it to a server instance."""

    def decorate(function):
        setattr(function, _DEFINITION_ATTR, ToolDefinition(name, options))
        return function

    return decorate


def register_tools(
    server: Any,
    modules: tuple[ModuleType, ...],
    *,
    disabled: set[str],
    descriptions: dict[str, str],
) -> set[str]:
    """Register a stable catalog, applying configuration before SDK registration."""
    definitions = {}
    for module in modules:
        for function in vars(module).values():
            definition = getattr(function, _DEFINITION_ATTR, None)
            if isinstance(definition, ToolDefinition):
                if definition.name in definitions:
                    raise ValueError(f"Duplicate tool name: {definition.name}")
                definitions[definition.name] = (function, definition)
    for name in sorted(definitions):
        if name in disabled:
            continue
        function, definition = definitions[name]
        options = dict(definition.options)
        if name in descriptions:
            options["description"] = descriptions[name]
        server.tool(name, **options)(function)
    return set(definitions)
