"""Utilities for extracting macro expansion info from Lean InfoTrees."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SyntaxRange(BaseModel):
    """Source code range for a syntax element."""

    start_line: int = Field(description="Start line (1-indexed)")
    start_col: int = Field(description="Start column (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    end_col: int = Field(description="End column (1-indexed)")
    synthetic: bool = Field(
        default=False, description="True if range is from macro-generated code"
    )


class MacroExpansion(BaseModel):
    """Information about a macro expansion."""

    original: str = Field(description="Original syntax before expansion")
    expanded: str = Field(description="Expanded syntax")
    syntax_kind: Optional[str] = Field(
        None, description="Syntax rule name (e.g., 'term_+_')"
    )
    referenced_constants: List[str] = Field(
        default_factory=list, description="Constants referenced in expansion"
    )
    range: Optional[SyntaxRange] = Field(None, description="Source range")
    nested_expansions: List["MacroExpansion"] = Field(
        default_factory=list,
        description="Nested macro expansions (e.g., double! 5 → 5+5 → HAdd.hAdd)",
    )


def _parse_range(range_info: Optional[Dict[str, Any]]) -> Optional[SyntaxRange]:
    """Parse range info from InfoTree node."""
    if not range_info:
        return None
    try:
        start = range_info.get("start", {})
        end = range_info.get("end", {})
        return SyntaxRange(
            start_line=start.get("line", 0),
            start_col=start.get("character", 0),
            end_line=end.get("line", 0),
            end_col=end.get("character", 0),
            synthetic=start.get("synthetic", False) or end.get("synthetic", False),
        )
    except Exception:
        return None


def _get_range_from_node(
    node: Dict[str, Any], max_depth: int = 3
) -> Optional[Dict[str, Any]]:
    """Get range from node or its descendants.

    MacroExpansion nodes often don't have range directly, but their
    children (usually Term nodes) do. Searches up to max_depth levels.
    """
    if node.get("range"):
        return node["range"]
    if max_depth <= 0:
        return None
    # Search children
    for child in node.get("children", []):
        range_info = _get_range_from_node(child, max_depth - 1)
        if range_info:
            return range_info
    return None


def _position_in_range(line: int, col: int, range_info: Optional[Dict]) -> bool:
    """Check if a position is within a range."""
    if not range_info:
        return False
    start = range_info.get("start", {})
    end = range_info.get("end", {})

    start_line = start.get("line", 0)
    start_col = start.get("character", 0)
    end_line = end.get("line", 0)
    end_col = end.get("character", 0)

    # Check if position is within range
    if line < start_line or line > end_line:
        return False
    if line == start_line and col < start_col:
        return False
    if line == end_line and col > end_col:
        return False
    return True


def _extract_constants(node: Dict[str, Any]) -> List[str]:
    """Extract constant names from Completion-Id descendant nodes.

    Parses entries like "HAdd.hAdd✝ : none" → "HAdd.hAdd"
    """
    constants = []
    for child in node.get("children", []):
        if child.get("type") == "Completion-Id":
            text = child.get("text", "")
            # Extract the identifier part before " : "
            if " : " in text:
                # Format: "[Completion-Id] HAdd.hAdd✝ : none @ ..."
                # Or just "HAdd.hAdd✝ : none"
                parts = text.split(" : ")
                if parts:
                    name = parts[0]
                    # Remove [Completion-Id] prefix if present
                    if name.startswith("[Completion-Id]"):
                        name = name[len("[Completion-Id]") :].strip()
                    # Remove the dagger symbol and trailing chars
                    name = name.rstrip("✝").rstrip()
                    # Only include qualified names (with dots) - these are constants
                    if "." in name:
                        constants.append(name)
        # Recurse into children
        constants.extend(_extract_constants(child))
    return list(set(constants))


def _collect_nested_expansions(children: List[Dict[str, Any]]) -> List[MacroExpansion]:
    """Recursively collect all MacroExpansion nodes from children.

    This handles chains like: double! 5 → 5 + 5 → HAdd.hAdd 5 5
    """
    nested = []
    for child in children:
        if _is_macro_expansion_node(child):
            extra = child.get("extra", "")
            if extra and "\n===>\n" in extra:
                parts = extra.split("\n===>\n")
                if len(parts) == 2:
                    # Recurse to get deeper nested expansions
                    deeper_nested = _collect_nested_expansions(
                        child.get("children", [])
                    )
                    # Get range from node or first child
                    range_info = _get_range_from_node(child)
                    nested.append(
                        MacroExpansion(
                            original=parts[0].strip(),
                            expanded=parts[1].strip(),
                            syntax_kind=child.get("elaborator"),
                            referenced_constants=_extract_constants(child),
                            range=_parse_range(range_info),
                            nested_expansions=deeper_nested,
                        )
                    )
        else:
            # Continue searching in non-MacroExpansion children
            nested.extend(_collect_nested_expansions(child.get("children", [])))
    return nested


def _is_macro_expansion_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a MacroExpansion node.

    MacroExpansion nodes can be identified by:
    1. type == "MacroExpansion" (sometimes set by parser)
    2. text starts with "Macro expansion" (raw format)
    """
    if node.get("type") == "MacroExpansion":
        return True
    text = node.get("text", "")
    if text.startswith("Macro expansion") or "[MacroExpansion]" in text:
        return True
    return False


def _find_expansion_at_position(
    node: Dict[str, Any], line: int, col: int
) -> Optional[MacroExpansion]:
    """Recursively search for MacroExpansion node containing position.

    Handles nested expansions: double! 5 → 5 + 5 → HAdd.hAdd 5 5
    by recursively collecting child MacroExpansion nodes.
    """
    if _is_macro_expansion_node(node):
        extra = node.get("extra", "")
        # Get range from node or first child (MacroExpansion nodes often lack range)
        range_info = _get_range_from_node(node)
        if extra and "\n===>\n" in extra and _position_in_range(line, col, range_info):
            parts = extra.split("\n===>\n")
            if len(parts) == 2:
                # Collect nested expansions from children
                nested = _collect_nested_expansions(node.get("children", []))

                return MacroExpansion(
                    original=parts[0].strip(),
                    expanded=parts[1].strip(),
                    syntax_kind=node.get("elaborator"),
                    referenced_constants=_extract_constants(node),
                    range=_parse_range(range_info),
                    nested_expansions=nested,
                )

    # Recurse into children
    for child in node.get("children", []):
        result = _find_expansion_at_position(child, line, col)
        if result:
            return result
    return None


def get_macro_expansion_at_position(
    trees: List[Dict[str, Any]], line: int, col: int
) -> Optional[MacroExpansion]:
    """Get macro expansion info at a position, including nested expansions.

    DEPRECATED: InfoTree positions have variable offsets from file positions.
    Use get_macro_expansion_by_text() instead for reliable matching.

    Args:
        trees: List of parsed InfoTree dictionaries from leanclient.get_info_trees()
        line: Line in Lean's internal format (varies based on file structure)
        col: Column in 0-indexed format

    Returns:
        MacroExpansion if found at position, None otherwise
    """
    for tree in trees:
        exp = _find_expansion_at_position(tree, line, col)
        if exp:
            return exp
    return None


def get_macro_expansion_by_text(
    trees: List[Dict[str, Any]], source_text: str
) -> Optional[MacroExpansion]:
    """Get macro expansion by matching source text.

    This is the reliable way to find macro expansions - by matching the
    source syntax text rather than positions (which have variable offsets).

    Args:
        trees: List of parsed InfoTree dictionaries from leanclient.get_info_trees()
        source_text: The source code text to find expansion for (from hover range)

    Returns:
        MacroExpansion if found matching the source text, None otherwise
    """
    source_text = source_text.strip()
    if not source_text:
        return None

    # Get all macro expansions and find one matching the source text
    for tree in trees:
        all_expansions = get_all_macro_expansions(tree)
        for exp in all_expansions:
            # Check if source_text matches the original (before expansion)
            orig = exp.original.strip()
            # Handle multi-line originals - first line often has the macro call
            first_line = orig.split("\n")[0].strip()

            if source_text == orig or source_text == first_line:
                return exp

            # Also try matching if source_text is contained in original
            if source_text in orig and len(source_text) >= 2:
                return exp

    return None


def get_all_macro_expansions(node: Dict[str, Any]) -> List[MacroExpansion]:
    """Get all macro expansions in a tree (for debugging/exploration).

    Args:
        node: Parsed InfoTree dictionary

    Returns:
        List of all MacroExpansion nodes found
    """
    expansions = []

    if _is_macro_expansion_node(node):
        extra = node.get("extra", "")
        if extra and "\n===>\n" in extra:
            parts = extra.split("\n===>\n")
            if len(parts) == 2:
                nested = _collect_nested_expansions(node.get("children", []))
                # Get range from node or first child
                range_info = _get_range_from_node(node)
                expansions.append(
                    MacroExpansion(
                        original=parts[0].strip(),
                        expanded=parts[1].strip(),
                        syntax_kind=node.get("elaborator"),
                        referenced_constants=_extract_constants(node),
                        range=_parse_range(range_info),
                        nested_expansions=nested,
                    )
                )

    # Recurse into children (but don't duplicate nested expansions)
    for child in node.get("children", []):
        expansions.extend(get_all_macro_expansions(child))

    return expansions
