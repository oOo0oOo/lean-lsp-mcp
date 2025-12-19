"""Tests for syntax_utils module - macro expansion extraction from InfoTrees."""

import pytest
from lean_lsp_mcp.syntax_utils import (
    MacroExpansion,
    SyntaxRange,
    get_macro_expansion_at_position,
    get_all_macro_expansions,
    _extract_constants,
    _find_expansion_at_position,
    _position_in_range,
    _parse_range,
)


class TestParseRange:
    def test_parse_range_basic(self):
        range_info = {
            "start": {"line": 10, "character": 5, "synthetic": False},
            "end": {"line": 10, "character": 15, "synthetic": False},
        }
        result = _parse_range(range_info)
        assert result is not None
        assert result.start_line == 10
        assert result.start_col == 5
        assert result.end_line == 10
        assert result.end_col == 15
        assert result.synthetic is False

    def test_parse_range_synthetic(self):
        range_info = {
            "start": {"line": 5, "character": 0, "synthetic": True},
            "end": {"line": 5, "character": 10, "synthetic": False},
        }
        result = _parse_range(range_info)
        assert result is not None
        assert result.synthetic is True

    def test_parse_range_none(self):
        assert _parse_range(None) is None
        assert _parse_range({}) is None


class TestPositionInRange:
    def test_position_inside_range(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(10, 10, range_info) is True

    def test_position_at_start(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(10, 5, range_info) is True

    def test_position_at_end(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(10, 15, range_info) is True

    def test_position_before_range(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(10, 4, range_info) is False

    def test_position_after_range(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(10, 16, range_info) is False

    def test_position_on_different_line(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 10, "character": 15},
        }
        assert _position_in_range(11, 10, range_info) is False

    def test_multiline_range(self):
        range_info = {
            "start": {"line": 10, "character": 5},
            "end": {"line": 12, "character": 15},
        }
        assert _position_in_range(11, 0, range_info) is True
        assert _position_in_range(10, 10, range_info) is True
        assert _position_in_range(12, 10, range_info) is True


class TestExtractConstants:
    def test_extract_constants_basic(self):
        node = {
            "children": [
                {
                    "type": "Completion-Id",
                    "text": "[Completion-Id] HAdd.hAdd : none",
                    "children": [],
                },
                {
                    "type": "Completion-Id",
                    "text": "[Completion-Id] Nat.add : some Type",
                    "children": [],
                },
            ]
        }
        constants = _extract_constants(node)
        assert "HAdd.hAdd" in constants
        assert "Nat.add" in constants

    def test_extract_constants_with_dagger(self):
        node = {
            "children": [
                {
                    "type": "Completion-Id",
                    "text": "[Completion-Id] HAdd.hAdd✝ : none",
                    "children": [],
                },
            ]
        }
        constants = _extract_constants(node)
        assert "HAdd.hAdd" in constants
        assert "HAdd.hAdd✝" not in constants

    def test_extract_constants_filters_local_vars(self):
        node = {
            "children": [
                {
                    "type": "Completion-Id",
                    "text": "[Completion-Id] n : none",  # Local var, no dot
                    "children": [],
                },
                {
                    "type": "Completion-Id",
                    "text": "[Completion-Id] List.map : some",  # Qualified name
                    "children": [],
                },
            ]
        }
        constants = _extract_constants(node)
        assert "List.map" in constants
        assert "n" not in constants

    def test_extract_constants_recursive(self):
        node = {
            "children": [
                {
                    "type": "Term",
                    "children": [
                        {
                            "type": "Completion-Id",
                            "text": "[Completion-Id] Eq.refl : some",
                            "children": [],
                        }
                    ],
                }
            ]
        }
        constants = _extract_constants(node)
        assert "Eq.refl" in constants


class TestFindExpansionAtPosition:
    def test_find_macro_expansion_basic(self):
        node = {
            "type": "MacroExpansion",
            "extra": "n + 0\n===>\nbinop% HAdd.hAdd n 0",
            "range": {
                "start": {"line": 10, "character": 5},
                "end": {"line": 10, "character": 10},
            },
            "elaborator": "term_+_",
            "children": [],
        }
        result = _find_expansion_at_position(node, 10, 7)
        assert result is not None
        assert result.original == "n + 0"
        assert "HAdd.hAdd" in result.expanded
        assert result.syntax_kind == "term_+_"

    def test_find_macro_expansion_not_at_position(self):
        node = {
            "type": "MacroExpansion",
            "extra": "n + 0\n===>\nbinop% HAdd.hAdd n 0",
            "range": {
                "start": {"line": 10, "character": 5},
                "end": {"line": 10, "character": 10},
            },
            "children": [],
        }
        result = _find_expansion_at_position(node, 5, 7)
        assert result is None

    def test_find_macro_expansion_nested(self):
        node = {
            "type": "Term",
            "children": [
                {
                    "type": "MacroExpansion",
                    "extra": "n + 0\n===>\nbinop% HAdd.hAdd n 0",
                    "range": {
                        "start": {"line": 10, "character": 5},
                        "end": {"line": 10, "character": 10},
                    },
                    "children": [],
                }
            ],
        }
        result = _find_expansion_at_position(node, 10, 7)
        assert result is not None
        assert result.original == "n + 0"


class TestGetMacroExpansionAtPosition:
    def test_get_macro_expansion_from_trees(self):
        trees = [
            {
                "type": "Command",
                "children": [
                    {
                        "type": "MacroExpansion",
                        "extra": "n + 0\n===>\nbinop% HAdd.hAdd n 0",
                        "range": {
                            "start": {"line": 10, "character": 5},
                            "end": {"line": 10, "character": 10},
                        },
                        "children": [],
                    }
                ],
            }
        ]
        result = get_macro_expansion_at_position(trees, 10, 7)
        assert result is not None
        assert result.original == "n + 0"

    def test_get_macro_expansion_empty_trees(self):
        result = get_macro_expansion_at_position([], 10, 7)
        assert result is None


class TestGetAllMacroExpansions:
    def test_get_all_expansions(self):
        node = {
            "type": "Command",
            "children": [
                {
                    "type": "MacroExpansion",
                    "extra": "n + 0\n===>\nbinop% HAdd.hAdd n 0",
                    "range": {
                        "start": {"line": 10, "character": 5},
                        "end": {"line": 10, "character": 10},
                    },
                    "children": [
                        {
                            "type": "MacroExpansion",
                            "extra": "n = n\n===>\nEq n n",
                            "range": {
                                "start": {"line": 10, "character": 12},
                                "end": {"line": 10, "character": 17},
                            },
                            "children": [],
                        }
                    ],
                }
            ],
        }
        expansions = get_all_macro_expansions(node)
        assert len(expansions) == 2  # Both top-level and nested
        originals = [e.original for e in expansions]
        assert "n + 0" in originals
        assert "n = n" in originals


class TestMacroExpansionModel:
    def test_macro_expansion_serialization(self):
        exp = MacroExpansion(
            original="n + 0",
            expanded="HAdd.hAdd n 0",
            syntax_kind="term_+_",
            referenced_constants=["HAdd.hAdd"],
            range=SyntaxRange(
                start_line=10, start_col=5, end_line=10, end_col=10, synthetic=False
            ),
            nested_expansions=[],
        )
        data = exp.model_dump()
        assert data["original"] == "n + 0"
        assert data["expanded"] == "HAdd.hAdd n 0"
        assert data["syntax_kind"] == "term_+_"
        assert "HAdd.hAdd" in data["referenced_constants"]
        assert data["range"]["start_line"] == 10

    def test_nested_macro_expansion(self):
        inner = MacroExpansion(
            original="5 + 5",
            expanded="HAdd.hAdd 5 5",
            referenced_constants=["HAdd.hAdd"],
        )
        outer = MacroExpansion(
            original="double! 5",
            expanded="5 + 5",
            syntax_kind="double!",
            nested_expansions=[inner],
        )
        assert outer.nested_expansions[0].original == "5 + 5"
        assert outer.nested_expansions[0].expanded == "HAdd.hAdd 5 5"
