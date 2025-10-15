from __future__ import annotations

import asyncio

from lean_lsp_mcp.utils import (
    OptionalTokenVerifier,
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


def test_extract_range_multiline() -> None:
    content = "alpha\nbeta"
    selection = {
        "start": {"line": 0, "character": 2},
        "end": {"line": 1, "character": 2},
    }

    assert extract_range(content, selection) == "pha\nbe"


def test_find_start_position() -> None:
    content = "foo\nbar baz"
    assert find_start_position(content, "bar") == {"line": 1, "column": 0}
    assert find_start_position(content, "missing") is None


def test_format_line_with_cursor() -> None:
    line = format_line("foo\nbar", 2, column=2)
    assert line == "b<cursor>ar"


def test_filter_diagnostics_by_position() -> None:
    diagnostics = [
        {
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 5},
            }
        }
    ]
    assert filter_diagnostics_by_position(diagnostics, 1, None) == diagnostics
    assert filter_diagnostics_by_position(diagnostics, 1, 3) == diagnostics
    assert filter_diagnostics_by_position(diagnostics, 1, 6) == []


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
