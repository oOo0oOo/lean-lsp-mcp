from __future__ import annotations

import asyncio

from lean_lsp_mcp.utils import (
    OptionalTokenVerifier,
    extract_goals_list,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_diagnostics,
    format_goal,
    format_line,
)


def test_format_diagnostics_compact_range() -> None:
    diagnostics = [
        {
            "fullRange": {
                "start": {"line": 3, "character": 1},
                "end": {"line": 3, "character": 5},
            },
            "severity": 2,
            "message": "Example message",
        }
    ]

    rendered = format_diagnostics(diagnostics)

    assert rendered == ["l4c2-l4c6, severity: 2\nExample message"]


def test_format_goal_strips_code_blocks() -> None:
    goal = {"rendered": "```lean\ntest\n```"}
    assert format_goal(goal, "fallback") == "test"
    assert format_goal(None, "fallback") == "fallback"


def test_extract_goals_list() -> None:
    # With goals list
    response = {"goals": ["goal1", "goal2"], "rendered": "..."}
    assert extract_goals_list(response) == ["goal1", "goal2"]

    # Empty goals list (proof complete)
    response = {"goals": [], "rendered": "no goals"}
    assert extract_goals_list(response) == []

    # None response (no goals at position)
    assert extract_goals_list(None) == []

    # Missing goals key
    assert extract_goals_list({"rendered": "..."}) == []


def test_extract_range_multiline() -> None:
    content = "alpha\nbeta"
    selection = {
        "start": {"line": 0, "character": 2},
        "end": {"line": 1, "character": 2},
    }

    assert extract_range(content, selection) == "pha\nbe"


def test_extract_range_handles_utf16_and_eof() -> None:
    content = "A😀B\n"
    selection = {
        "start": {"line": 0, "character": 1},
        "end": {"line": 1, "character": 0},
    }

    assert extract_range(content, selection) == "😀B\n"


def test_find_start_position() -> None:
    content = "foo\nbar baz"
    assert find_start_position(content, "bar") == {"line": 1, "column": 0}
    assert find_start_position(content, "missing") is None


def test_format_line_with_cursor() -> None:
    line = format_line("foo\nbar", 2, column=2)
    assert line == "b<cursor>ar"


def test_filter_diagnostics_by_position() -> None:
    def make_range(
        start_line: int,
        start_char: int | None,
        end_line: int,
        end_char: int | None,
    ) -> dict:
        start = {"line": start_line}
        if start_char is not None:
            start["character"] = start_char
        end = {"line": end_line}
        if end_char is not None:
            end["character"] = end_char
        return {"range": {"start": start, "end": end}}

    diag_same_line = make_range(1, 0, 1, 5)
    diag_multiline = make_range(0, 2, 1, 0)
    diag_point = make_range(2, 3, 2, 3)
    diag_missing_start = make_range(4, None, 4, 5)

    diagnostics = [diag_same_line, diag_multiline, diag_point, diag_missing_start]

    # No line filtering returns a copy of all diagnostics
    result_all = filter_diagnostics_by_position(diagnostics, None, None)
    assert result_all == diagnostics
    assert result_all is not diagnostics

    # Same line range selections
    assert filter_diagnostics_by_position(diagnostics, 1, None) == [diag_same_line]
    assert filter_diagnostics_by_position(diagnostics, 1, 3) == [diag_same_line]
    assert filter_diagnostics_by_position(diagnostics, 1, 6) == []

    # Multiline diagnostic shouldn't match trailing zero-width end on next line
    assert filter_diagnostics_by_position(diagnostics, 0, None) == [diag_multiline]
    assert filter_diagnostics_by_position(diagnostics, 1, 0) == [diag_same_line]

    # Point diagnostic requires exact column match
    assert filter_diagnostics_by_position(diagnostics, 2, 3) == [diag_point]
    assert filter_diagnostics_by_position(diagnostics, 2, 2) == []

    # Missing start character defaults to column zero
    assert filter_diagnostics_by_position(diagnostics, 4, 1) == [diag_missing_start]
    assert filter_diagnostics_by_position(diagnostics, 4, 5) == []


def test_optional_token_verifier() -> None:
    verifier = OptionalTokenVerifier("secret")
    granted = asyncio.run(verifier.verify_token("secret"))
    rejected = asyncio.run(verifier.verify_token("nope"))

    assert granted is not None
    assert granted.token == "secret"
    assert rejected is None


def test_format_diagnostics_line_filter() -> None:
    diagnostics = [
        {
            "fullRange": {
                "start": {"line": 2, "character": 0},
                "end": {"line": 2, "character": 3},
            },
            "range": {
                "start": {"line": 2, "character": 0},
                "end": {"line": 2, "character": 3},
            },
            "severity": 1,
            "message": "Only on line three",
        }
    ]

    keep_all = format_diagnostics(diagnostics, select_line=-1)
    only_line_two = format_diagnostics(diagnostics, select_line=2)
    other_line = format_diagnostics(diagnostics, select_line=1)

    assert keep_all == ["l3c1-l3c4, severity: 1\nOnly on line three"]
    assert only_line_two == ["l3c1-l3c4, severity: 1\nOnly on line three"]
    assert other_line == []
