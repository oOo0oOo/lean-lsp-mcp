from __future__ import annotations

from lean_lsp_mcp.server import _extract_try_this


def test_extract_try_this_single_line() -> None:
    msg = "Some error. Try this: exact h"
    assert _extract_try_this(msg) == ["exact h"]


def test_extract_try_this_multiline() -> None:
    msg = "Try this:\n  simp\n  ring"
    assert _extract_try_this(msg) == ["simp", "ring"]
