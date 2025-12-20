"""Tree visualization for macro expansion chains.

Generates SVG diagrams of MacroExpansion trees using Graphviz.
"""

import base64
import html
from typing import Optional

from lean_lsp_mcp.syntax_utils import MacroExpansion

# Dark theme colors (VS Code style)
COLORS = {
    "bg": "#1e1e1e",
    "node_bg": "#2d2d2d",
    "node_border": "#3c3c3c",
    "text": "#d4d4d4",
    "syntax_kind": "#569cd6",  # blue
    "arrow": "#808080",
    "refs": "#6a9955",  # green
}


def _escape_dot_label(text: str) -> str:
    """Escape text for use in DOT labels."""
    # Escape special characters for DOT
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    # Truncate long text
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def _escape_html_label(text: str) -> str:
    """Escape text for use in HTML-like DOT labels."""
    text = html.escape(text)
    # Truncate long text
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def _build_node_label(expansion: MacroExpansion, is_leaf: bool = False) -> str:
    """Build an HTML-like label for a node.

    Shows:
    - original syntax (main text)
    - syntax_kind in blue (if present)
    - referenced_constants in green (if leaf and present)
    """
    parts = []

    # Main text: the original syntax
    original = _escape_html_label(expansion.original)
    parts.append(f'<b>{original}</b>')

    # Syntax kind in blue
    if expansion.syntax_kind:
        kind = _escape_html_label(expansion.syntax_kind)
        parts.append(f'<font color="{COLORS["syntax_kind"]}">{kind}</font>')

    # Referenced constants in green (only for leaf nodes)
    if is_leaf and expansion.referenced_constants:
        refs = ", ".join(expansion.referenced_constants[:3])
        if len(expansion.referenced_constants) > 3:
            refs += "..."
        refs = _escape_html_label(refs)
        parts.append(f'<font color="{COLORS["refs"]}">{refs}</font>')

    return "<" + "<br/>".join(parts) + ">"


def macro_expansion_to_dot(expansion: MacroExpansion) -> str:
    """Convert MacroExpansion tree to Graphviz DOT format with dark theme.

    Args:
        expansion: The macro expansion tree to visualize

    Returns:
        DOT format string representing the tree
    """
    lines = [
        "digraph MacroExpansion {",
        "    rankdir=TB;",
        f'    bgcolor="{COLORS["bg"]}";',
        f'    node [shape=box, style="rounded,filled", fillcolor="{COLORS["node_bg"]}",',
        f'          color="{COLORS["node_border"]}", fontcolor="{COLORS["text"]}",',
        '          fontname="Consolas, Monaco, monospace", fontsize=11];',
        f'    edge [color="{COLORS["arrow"]}", arrowsize=0.8];',
        "",
    ]

    node_id = [0]  # Mutable counter for unique node IDs

    def add_node(exp: MacroExpansion, parent_id: Optional[int] = None) -> int:
        """Recursively add nodes and edges to the graph."""
        current_id = node_id[0]
        node_id[0] += 1

        is_leaf = len(exp.nested_expansions) == 0
        label = _build_node_label(exp, is_leaf)
        lines.append(f"    n{current_id} [label={label}];")

        if parent_id is not None:
            lines.append(f"    n{parent_id} -> n{current_id};")

        # Add expanded form as intermediate node if different from nested
        if exp.expanded and exp.expanded != exp.original:
            # If there are nested expansions, the expanded form is implicit
            # Only show it explicitly if it's a leaf
            if is_leaf:
                expanded_id = node_id[0]
                node_id[0] += 1
                expanded_label = _escape_html_label(exp.expanded)
                lines.append(
                    f'    n{expanded_id} [label=<{expanded_label}>, '
                    f'fillcolor="{COLORS["node_bg"]}", style="rounded,filled,dashed"];'
                )
                lines.append(f"    n{current_id} -> n{expanded_id};")

        # Recurse into nested expansions
        for nested in exp.nested_expansions:
            add_node(nested, current_id)

        return current_id

    add_node(expansion)

    lines.append("}")
    return "\n".join(lines)


def render_expansion_svg(expansion: MacroExpansion) -> str:
    """Render MacroExpansion as SVG string.

    Args:
        expansion: The macro expansion tree to visualize

    Returns:
        SVG string of the rendered diagram

    Raises:
        ImportError: If graphviz is not installed
        RuntimeError: If graphviz rendering fails
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package required for visualization. "
            "Install with: pip install lean-lsp-mcp[viz]"
        )

    dot_source = macro_expansion_to_dot(expansion)

    # Create a graph from DOT source
    graph = graphviz.Source(dot_source)

    # Render to SVG
    try:
        svg_bytes = graph.pipe(format="svg")
        return svg_bytes.decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to render SVG: {e}") from e


def render_expansion_svg_base64(expansion: MacroExpansion) -> str:
    """Render MacroExpansion as base64-encoded SVG.

    Args:
        expansion: The macro expansion tree to visualize

    Returns:
        Base64-encoded SVG string suitable for data URLs
    """
    svg = render_expansion_svg(expansion)
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def render_expansion_png(expansion: MacroExpansion) -> bytes:
    """Render MacroExpansion as PNG bytes.

    Args:
        expansion: The macro expansion tree to visualize

    Returns:
        PNG image bytes

    Raises:
        ImportError: If graphviz is not installed
        RuntimeError: If graphviz rendering fails
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package required for visualization. "
            "Install with: pip install lean-lsp-mcp[viz]"
        )

    dot_source = macro_expansion_to_dot(expansion)
    graph = graphviz.Source(dot_source)

    try:
        return graph.pipe(format="png")
    except Exception as e:
        raise RuntimeError(f"Failed to render PNG: {e}") from e


def render_expansion_png_base64(expansion: MacroExpansion) -> str:
    """Render MacroExpansion as base64-encoded PNG.

    Args:
        expansion: The macro expansion tree to visualize

    Returns:
        Base64-encoded PNG string suitable for data URLs
    """
    png_bytes = render_expansion_png(expansion)
    return base64.b64encode(png_bytes).decode("ascii")
