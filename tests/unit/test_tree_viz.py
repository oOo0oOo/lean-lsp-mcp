"""Tests for tree_viz module - macro expansion visualization."""

import pytest

from lean_lsp_mcp.syntax_utils import MacroExpansion
from lean_lsp_mcp.tree_viz import (
    COLORS,
    _build_node_label,
    _escape_dot_label,
    _escape_html_label,
    macro_expansion_to_dot,
)


class TestEscaping:
    """Tests for text escaping functions."""

    def test_escape_dot_label_basic(self):
        """Basic text passes through."""
        assert _escape_dot_label("hello") == "hello"

    def test_escape_dot_label_quotes(self):
        """Quotes are escaped."""
        assert _escape_dot_label('say "hello"') == 'say \\"hello\\"'

    def test_escape_dot_label_backslash(self):
        """Backslashes are escaped."""
        assert _escape_dot_label("a\\b") == "a\\\\b"

    def test_escape_dot_label_newline(self):
        """Newlines are escaped."""
        assert _escape_dot_label("a\nb") == "a\\nb"

    def test_escape_dot_label_truncation(self):
        """Long text is truncated."""
        long_text = "x" * 100
        result = _escape_dot_label(long_text)
        assert len(result) == 60
        assert result.endswith("...")

    def test_escape_html_label_entities(self):
        """HTML entities are escaped."""
        assert _escape_html_label("<div>") == "&lt;div&gt;"
        assert _escape_html_label("a & b") == "a &amp; b"

    def test_escape_html_label_truncation(self):
        """Long HTML text is truncated."""
        long_text = "x" * 100
        result = _escape_html_label(long_text)
        assert len(result) == 60
        assert result.endswith("...")


class TestNodeLabel:
    """Tests for node label building."""

    def test_simple_expansion(self):
        """Simple expansion shows original text."""
        exp = MacroExpansion(original="center", expanded="b2")
        label = _build_node_label(exp, is_leaf=True)
        assert "<b>center</b>" in label

    def test_with_syntax_kind(self):
        """Syntax kind is shown in blue."""
        exp = MacroExpansion(original="double! 5", expanded="5 + 5", syntax_kind="double!")
        label = _build_node_label(exp, is_leaf=False)
        assert "double!" in label
        assert COLORS["syntax_kind"] in label

    def test_with_constants_leaf(self):
        """Referenced constants shown on leaf nodes."""
        exp = MacroExpansion(
            original="5 + 5",
            expanded="HAdd.hAdd 5 5",
            referenced_constants=["HAdd.hAdd"],
        )
        label = _build_node_label(exp, is_leaf=True)
        assert "HAdd.hAdd" in label
        assert COLORS["refs"] in label

    def test_with_constants_non_leaf(self):
        """Referenced constants not shown on non-leaf nodes."""
        exp = MacroExpansion(
            original="5 + 5",
            expanded="HAdd.hAdd 5 5",
            referenced_constants=["HAdd.hAdd"],
        )
        label = _build_node_label(exp, is_leaf=False)
        assert "HAdd.hAdd" not in label

    def test_many_constants_truncated(self):
        """Many constants are truncated."""
        exp = MacroExpansion(
            original="expr",
            expanded="...",
            referenced_constants=["A", "B", "C", "D", "E"],
        )
        label = _build_node_label(exp, is_leaf=True)
        assert "A, B, C..." in label


class TestDotGeneration:
    """Tests for DOT format generation."""

    def test_simple_expansion_dot(self):
        """Simple expansion generates valid DOT."""
        exp = MacroExpansion(original="center", expanded="b2")
        dot = macro_expansion_to_dot(exp)

        assert "digraph MacroExpansion" in dot
        assert "rankdir=TB" in dot
        assert COLORS["bg"] in dot
        assert "center" in dot

    def test_nested_expansion_dot(self):
        """Nested expansion generates connected nodes."""
        inner = MacroExpansion(original="b2", expanded="(4 : Pos)")
        outer = MacroExpansion(
            original="center", expanded="b2", nested_expansions=[inner]
        )
        dot = macro_expansion_to_dot(outer)

        # Should have edges
        assert "->" in dot
        # Both nodes should exist
        assert "center" in dot
        assert "b2" in dot

    def test_deeply_nested_expansion(self):
        """Deeply nested expansion generates full tree."""
        level3 = MacroExpansion(original="(4 : Pos)", expanded="Pos.mk 4")
        level2 = MacroExpansion(original="b2", expanded="(4 : Pos)", nested_expansions=[level3])
        level1 = MacroExpansion(original="center", expanded="b2", nested_expansions=[level2])

        dot = macro_expansion_to_dot(level1)

        # Count edges - should have at least 2
        edge_count = dot.count("->")
        assert edge_count >= 2

    def test_dark_theme_colors(self):
        """DOT uses dark theme colors."""
        exp = MacroExpansion(original="test", expanded="result")
        dot = macro_expansion_to_dot(exp)

        assert COLORS["bg"] in dot
        assert COLORS["node_bg"] in dot
        assert COLORS["node_border"] in dot
        assert COLORS["text"] in dot

    def test_html_entities_escaped(self):
        """HTML entities in labels are escaped."""
        exp = MacroExpansion(original="a < b", expanded="Less.lt a b")
        dot = macro_expansion_to_dot(exp)

        # Should be escaped
        assert "&lt;" in dot


class TestSvgRendering:
    """Tests for SVG rendering (requires graphviz)."""

    @pytest.fixture
    def simple_expansion(self):
        return MacroExpansion(original="center", expanded="b2")

    @pytest.fixture
    def nested_expansion(self):
        inner = MacroExpansion(original="b2", expanded="(4 : Pos)")
        return MacroExpansion(original="center", expanded="b2", nested_expansions=[inner])

    def test_render_svg_import_error(self, simple_expansion):
        """render_expansion_svg raises ImportError if graphviz not installed."""
        # This test checks the error handling when graphviz is not available
        # In practice, graphviz should be installed for this test suite
        from lean_lsp_mcp.tree_viz import render_expansion_svg

        try:
            svg = render_expansion_svg(simple_expansion)
            # If graphviz is installed, we should get valid SVG
            assert "<?xml" in svg or "<svg" in svg
        except ImportError as e:
            assert "graphviz" in str(e).lower()
        except RuntimeError as e:
            # Graphviz binary not found - this is also valid
            assert "render" in str(e).lower() or "graphviz" in str(e).lower()

    def test_render_svg_base64(self, simple_expansion):
        """render_expansion_svg_base64 returns base64 encoded SVG."""
        from lean_lsp_mcp.tree_viz import render_expansion_svg_base64
        import base64

        try:
            b64 = render_expansion_svg_base64(simple_expansion)
            # Should be valid base64
            decoded = base64.b64decode(b64)
            assert b"<svg" in decoded or b"<?xml" in decoded
        except (ImportError, RuntimeError):
            pytest.skip("graphviz not available")

    def test_render_png(self, simple_expansion):
        """render_expansion_png returns PNG bytes."""
        from lean_lsp_mcp.tree_viz import render_expansion_png

        try:
            png_bytes = render_expansion_png(simple_expansion)
            # PNG magic bytes
            assert png_bytes[:4] == b"\x89PNG"
        except (ImportError, RuntimeError):
            pytest.skip("graphviz not available")


class TestRealExpansion:
    """Tests with realistic expansion structures."""

    def test_tictactoe_center_expansion(self):
        """TicTacToe center macro expansion."""
        # center -> b2 -> (4 : Pos)
        level2 = MacroExpansion(
            original="(4 : Pos)",
            expanded="Pos.mk 4",
            referenced_constants=["Pos.mk"],
        )
        level1 = MacroExpansion(
            original="b2",
            expanded="(4 : Pos)",
            syntax_kind="term",
            nested_expansions=[level2],
        )
        root = MacroExpansion(
            original="center",
            expanded="b2",
            syntax_kind="term",
            nested_expansions=[level1],
        )

        dot = macro_expansion_to_dot(root)

        assert "center" in dot
        assert "b2" in dot
        assert "Pos" in dot
        assert "->" in dot  # Has connections

    def test_game_sequence_expansion(self):
        """Game sequence ⟪a1, b2, c3⟫ expansion."""
        # Multiple nested expansions
        a1 = MacroExpansion(original="a1", expanded="(0 : Pos)")
        b2 = MacroExpansion(original="b2", expanded="(4 : Pos)")
        c3 = MacroExpansion(original="c3", expanded="(8 : Pos)")

        seq = MacroExpansion(
            original="⟪a1, b2, c3⟫",
            expanded="playSequence [a1, b2, c3]",
            syntax_kind="game_seq",
            nested_expansions=[a1, b2, c3],
        )

        dot = macro_expansion_to_dot(seq)

        # All positions should be in the graph
        assert "a1" in dot
        assert "b2" in dot
        assert "c3" in dot
        assert "playSequence" in dot or "game_seq" in dot
