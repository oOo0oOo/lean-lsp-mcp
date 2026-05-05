"""Unit tests for the minimal_hypotheses parser helpers."""

from __future__ import annotations

from lean_lsp_mcp.minimal_hypotheses import (
    _find_balanced_close,
    drop_binder,
    explicit_hypotheses,
    find_theorem_binders,
    line_of_offset,
)


class TestFindBalancedClose:
    def test_simple(self):
        assert _find_balanced_close("(a)", 0, "(", ")") == 2

    def test_nested(self):
        assert _find_balanced_close("((x)y)", 0, "(", ")") == 5
        assert _find_balanced_close("((x)y)", 1, "(", ")") == 3

    def test_unbalanced_returns_negative(self):
        assert _find_balanced_close("((x)", 0, "(", ")") == -1


class TestFindTheoremBinders:
    def test_returns_each_binder_in_order(self):
        src = (
            "theorem foo (h1 : P) (h2 : Q) {x : Nat} [DecidableEq α] : R := by\n"
            "  exact h1\n"
        )
        texts = [b[0] for b in find_theorem_binders(src, "foo")]
        assert texts == ["(h1 : P)", "(h2 : Q)", "{x : Nat}", "[DecidableEq α]"]

    def test_handles_nested_parens_in_type(self):
        src = "theorem foo (h : P → (Q ∧ R)) (k : S) : T := sorry\n"
        texts = [b[0] for b in find_theorem_binders(src, "foo")]
        assert texts == ["(h : P → (Q ∧ R))", "(k : S)"]

    def test_no_binders(self):
        assert (
            find_theorem_binders("theorem trivial : True := trivial\n", "trivial") == []
        )

    def test_unknown_name(self):
        assert find_theorem_binders("theorem foo (h : P) : Q := sorry\n", "bar") == []

    def test_lemma_keyword(self):
        src = "lemma bar (h : 1 = 1) : True := trivial\n"
        texts = [b[0] for b in find_theorem_binders(src, "bar")]
        assert texts == ["(h : 1 = 1)"]


class TestExplicitHypotheses:
    def test_filters_out_implicit_and_instance(self):
        src = "theorem foo (h : P) {x : Nat} [DecidableEq α] : Q := sorry"
        binders = find_theorem_binders(src, "foo")
        explicit = explicit_hypotheses(binders)
        assert [b[0] for b in explicit] == ["(h : P)"]

    def test_only_keeps_colon_paren_groups(self):
        src = "theorem foo (h : P) (a b c : Nat) (no_colon_here) : Q := sorry"
        binders = find_theorem_binders(src, "foo")
        explicit = [b[0] for b in explicit_hypotheses(binders)]
        # Last group has no colon, so it's not treated as a hypothesis.
        assert explicit == ["(h : P)", "(a b c : Nat)"]


class TestDropBinder:
    def test_removes_binder_and_one_leading_space(self):
        src = "theorem foo (h1 : P) (h2 : Q) : R := sorry"
        binders = find_theorem_binders(src, "foo")
        # Drop h2
        _, start, end = binders[1]
        out = drop_binder(src, start, end)
        assert out == "theorem foo (h1 : P) : R := sorry"

    def test_first_binder_with_no_leading_space_is_safe(self):
        # Synthetic: drop a binder at offset 0 (would not happen in real Lean
        # source but the helper shouldn't crash).
        out = drop_binder("(h : P) rest", 0, 7)
        assert out == " rest"


def test_line_of_offset_one_indexed():
    src = "a\nbb\nccc\n"
    assert line_of_offset(src, 0) == 1
    assert line_of_offset(src, 2) == 2
    assert line_of_offset(src, 5) == 3
