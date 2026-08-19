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


def register_module_tools(server: Any, module: ModuleType) -> None:
    """Register every tool declared in ``module`` on ``server``."""
    for value in vars(module).values():
        definition = getattr(value, _DEFINITION_ATTR, None)
        if isinstance(definition, ToolDefinition):
            server.tool(definition.name, **definition.options)(value)
