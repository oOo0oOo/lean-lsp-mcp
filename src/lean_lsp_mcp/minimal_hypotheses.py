"""Minimal-hypothesis pruning: parse a theorem's binders, drop each
explicit `(h : T)` in turn, and let the caller re-check via the LSP.

Pure parsing utilities — no I/O, no LSP. Designed so that callers (server.py)
can drive the LSP overlay loop without re-implementing the binder parser.
"""

from __future__ import annotations

import re

_DECL_RE = re.compile(r"\b(theorem|lemma|example|def)\s+([A-Za-z_][A-Za-z0-9_'.]*)")


def _find_balanced_close(s: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Index of the matching close char, or -1 if unbalanced.
    Counts opens and closes only; does not skip strings or comments."""
    depth = 0
    for i in range(open_pos, len(s)):
        ch = s[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_theorem_binders(source: str, name: str) -> list[tuple[str, int, int]]:
    """Find the binder list of a theorem/lemma/def.

    Walks forward from the declaration name, consuming consecutive top-level
    `(...)`, `{...}`, and `[...]` groups until it hits the type-introducing `:`
    or end-of-binders. Returns `[(binder_text, start, end_exclusive), ...]` in
    source order.

    Note: doesn't strip Lean comments — assume well-formatted code. Matches the
    *first* declaration in `source` whose name equals `name`; bare names should
    be unique within a file or this will pick the wrong one.
    """
    pattern = re.compile(rf"\b(theorem|lemma|example|def)\s+{re.escape(name)}\b")
    m = pattern.search(source)
    if not m:
        return []

    pos = m.end()
    binders: list[tuple[str, int, int]] = []
    closers = {"(": ")", "{": "}", "[": "]"}
    while pos < len(source):
        while pos < len(source) and source[pos] in " \t\n\r":
            pos += 1
        if pos >= len(source):
            break
        ch = source[pos]
        if ch in closers:
            close = _find_balanced_close(source, pos, ch, closers[ch])
            if close == -1:
                break
            binders.append((source[pos : close + 1], pos, close + 1))
            pos = close + 1
        else:
            break
    return binders


def explicit_hypotheses(
    binders: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Filter a binder list to just `(h : T)` style explicit hypotheses.

    Skips implicit `{x : α}` and instance `[inst : C]` binders — those are
    inferable / always load-bearing in practice.
    """
    return [(b, s, e) for (b, s, e) in binders if b.startswith("(") and ":" in b]


def drop_binder(source: str, start: int, end: int) -> str:
    """Return `source` with the binder at [start, end) removed,
    also absorbing one preceding space so two single-spaces don't collide."""
    cut = start - 1 if start > 0 and source[start - 1] == " " else start
    return source[:cut] + source[end:]


def line_of_offset(source: str, offset: int) -> int:
    """1-indexed line number of a character offset."""
    return source.count("\n", 0, offset) + 1
