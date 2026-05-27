"""Regression tests for the multi_attempt LSP-path false-success fix.

When a snippet introduced via the LSP path triggered an error at a line
outside the local edit range (e.g. a `whnf` heartbeat timeout reported
at a distant declaration), the line-range filter discarded the
diagnostic and the result became goals=[]/diagnostics=[], indistinguishable
from genuine tactic success. The fix snapshots baseline diagnostics
before the edit and unions any *new* diagnostic into the result
regardless of its line position.
"""

from __future__ import annotations

from lean_lsp_mcp.server import (
    _diagnostic_identity,
    _filter_diagnostics_by_line_range,
    _prepare_multi_attempt_edit,
    _shift_baseline_keys,
)


def _diag(line: int, char: int, severity: int, message: str) -> dict:
    return {
        "range": {
            "start": {"line": line, "character": char},
            "end": {"line": line, "character": char + 1},
        },
        "severity": severity,
        "message": message,
    }


def test_identity_is_stable_across_equal_dicts() -> None:
    a = _diag(line=10, char=0, severity=1, message="x")
    b = _diag(line=10, char=0, severity=1, message="x")
    assert _diagnostic_identity(a) == _diagnostic_identity(b)


def test_identity_distinguishes_by_message() -> None:
    a = _diag(line=10, char=0, severity=1, message="x")
    b = _diag(line=10, char=0, severity=1, message="y")
    assert _diagnostic_identity(a) != _diagnostic_identity(b)


def test_identity_distinguishes_by_line() -> None:
    a = _diag(line=10, char=0, severity=1, message="x")
    b = _diag(line=11, char=0, severity=1, message="x")
    assert _diagnostic_identity(a) != _diagnostic_identity(b)


def test_identity_handles_missing_range() -> None:
    d = {"severity": 1, "message": "no range"}
    # Should produce a valid identity tuple, not raise.
    ident = _diagnostic_identity(d)
    # (start.line, start.char, end.line, end.char, severity, code, source, message)
    assert ident == (None, None, None, None, 1, None, None, "no range")


def test_identity_distinguishes_by_code_and_source() -> None:
    """Two diagnostics with the same range/severity/message but different
    ``code`` or ``source`` (e.g. one from the elaborator, one from a
    linter) must produce different identities.
    """
    base = _diag(line=10, char=0, severity=2, message="unused")
    linter = dict(base, code="linter.unusedVariables", source="lint")
    elab = dict(base, code=None, source=None)
    assert _diagnostic_identity(linter) != _diagnostic_identity(elab)


def test_new_diagnostic_outside_range_survives_set_diff() -> None:
    """Bug scenario: snippet introduces an error at a distant line, the
    line-range filter discards it, but the baseline-diff catches it.
    """
    baseline = [_diag(line=20, char=0, severity=2, message="declaration uses `sorry`")]
    after_snippet = [
        _diag(line=20, char=0, severity=2, message="declaration uses `sorry`"),
        _diag(line=5, char=0, severity=1, message="deterministic timeout at `whnf`"),
    ]
    baseline_keys = {_diagnostic_identity(d) for d in baseline}

    # Snippet inserted at line 21; local filter window is line 20..21.
    filtered = _filter_diagnostics_by_line_range(after_snippet, 20, 21)
    in_filtered = {id(d) for d in filtered}
    extra = [
        d
        for d in after_snippet
        if id(d) not in in_filtered and _diagnostic_identity(d) not in baseline_keys
    ]

    # The original `sorry` warning is in filtered (overlaps line range).
    assert any("sorry" in d["message"] for d in filtered)
    # The distant whnf error must be picked up by the set-diff path.
    assert any("whnf" in d["message"] for d in extra)


def test_unchanged_distant_diagnostic_not_re_reported() -> None:
    """Pre-existing diagnostics outside the snippet range should NOT pollute
    the result if they were already there before the edit.
    """
    pre_existing = _diag(line=200, char=0, severity=1, message="pre-existing error")
    baseline = [pre_existing]
    # After snippet, the same pre-existing error is still there (no change).
    after_snippet = [pre_existing]
    baseline_keys = {_diagnostic_identity(d) for d in baseline}

    filtered = _filter_diagnostics_by_line_range(after_snippet, 10, 11)
    in_filtered = {id(d) for d in filtered}
    extra = [
        d
        for d in after_snippet
        if id(d) not in in_filtered and _diagnostic_identity(d) not in baseline_keys
    ]

    assert filtered == []  # outside range
    assert extra == []  # in baseline, so excluded


def test_prepare_multi_attempt_edit_no_shift_in_middle_of_file() -> None:
    """Multi-line snippet replacing N source lines in the middle of a file
    leaves the file size unchanged, so line_delta == 0.
    """
    _, _, _, _, line_delta = _prepare_multi_attempt_edit(
        line_context="  sorry",
        target_column=2,
        snippet="have h : True := trivial\n  exact h",
        total_lines=100,
        line=50,
    )
    assert line_delta == 0


def test_prepare_multi_attempt_edit_shift_when_clamped() -> None:
    """Multi-line snippet near end-of-file gets its replacement range clamped
    by total_lines. The payload still has N lines but only fewer original
    lines were replaced, so the file grows by the difference.
    """
    # 4-line file, snippet at line 3 with 3 payload lines.
    # replacement covers lines [2, min(2+3, 4)) = [2, 4), i.e. 2 original
    # lines. Payload is 3 lines. Delta = +1.
    _, _, _, _, line_delta = _prepare_multi_attempt_edit(
        line_context="  sorry",
        target_column=2,
        snippet="have h : True := trivial\n  have h2 : True := h\n  exact h2",
        total_lines=4,
        line=3,
    )
    assert line_delta == 1


def test_shift_baseline_keys_no_op_for_zero_delta() -> None:
    keys = {_diagnostic_identity(_diag(line=10, char=0, severity=2, message="x"))}
    shifted = _shift_baseline_keys(keys, edit_start_line=5, line_delta=0)
    assert shifted is keys


def test_shift_baseline_keys_shifts_only_at_or_after_edit() -> None:
    """Pre-existing diagnostics BEFORE the edit are unaffected; those
    AT/AFTER the edit shift by line_delta.
    """
    before = _diagnostic_identity(_diag(line=2, char=0, severity=2, message="before"))
    after = _diagnostic_identity(_diag(line=8, char=0, severity=2, message="after"))
    shifted = _shift_baseline_keys({before, after}, edit_start_line=5, line_delta=2)

    # `before` stays at line 2.
    before_in_shifted = next(k for k in shifted if k[7] == "before")
    assert before_in_shifted[0] == 2
    # `after` was at line 8, end at line 8 too; both shift by +2.
    after_in_shifted = next(k for k in shifted if k[7] == "after")
    assert after_in_shifted[0] == 10
    assert after_in_shifted[2] == 10


def test_shifted_baseline_eliminates_false_positive_from_line_shift() -> None:
    """End-to-end: pre-existing diagnostic at line 5 shifts to line 7 after
    a 2-line file growth from a multi-line snippet. Without the shift fix,
    the post-edit line-7 diagnostic would be reported as 'new'. With the
    shift fix, the shifted baseline key matches it and it's correctly
    excluded from extra_diag.
    """
    # Baseline: pre-existing "uses sorry" warning at line 5.
    baseline = [_diag(line=5, char=0, severity=2, message="declaration uses `sorry`")]
    # Edit at line 3 (0-indexed 2), file grows by 2. Pre-existing line-5
    # diagnostic gets re-emitted at line 7 post-edit.
    after_snippet = [
        _diag(line=7, char=0, severity=2, message="declaration uses `sorry`")
    ]
    baseline_keys = {_diagnostic_identity(d) for d in baseline}

    # Without shift adjustment: the post-edit diag is treated as new — bug.
    filtered = _filter_diagnostics_by_line_range(after_snippet, 2, 4)
    in_filtered = {id(d) for d in filtered}
    unshifted_extra = [
        d
        for d in after_snippet
        if id(d) not in in_filtered and _diagnostic_identity(d) not in baseline_keys
    ]
    assert unshifted_extra, "without the fix, the shifted baseline diag false-positives"

    # With shift adjustment: extra_diag is empty.
    shifted = _shift_baseline_keys(baseline_keys, edit_start_line=2, line_delta=2)
    shifted_extra = [
        d
        for d in after_snippet
        if id(d) not in in_filtered and _diagnostic_identity(d) not in shifted
    ]
    assert shifted_extra == []
