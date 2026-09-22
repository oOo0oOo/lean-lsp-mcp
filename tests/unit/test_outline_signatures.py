from __future__ import annotations

from lean_lsp_mcp.outline_utils import _body_assignment_index, _extract_declarations


def _signature(source: str) -> str | None:
    declarations = _extract_declarations(source, 0, len(source.splitlines()))
    assert len(declarations) == 1, declarations
    return declarations[0]["_type"]


def test_an_ordinary_statement_splits_at_its_only_assignment() -> None:
    assert _body_assignment_index("theorem t : True := trivial") == 17


def test_a_statement_with_no_body_has_no_split() -> None:
    assert _body_assignment_index("theorem t : True") is None


def test_a_let_bound_statement_keeps_its_conclusion() -> None:
    """Each `let` in a statement spends one `:=` before the body's.

    Splitting on the first one cuts the type at the binder, hiding the
    conclusion -- which is the part an outline exists to show.
    """
    source = "\n".join(
        [
            "theorem bound (n : Nat) :",
            "    let m := n + 1",
            "    m > n := by",
            "  simp",
        ]
    )

    signature = _signature(source)

    assert signature is not None
    assert "m > n" in signature, signature
    assert signature.endswith("m > n")


def test_several_binders_are_all_skipped() -> None:
    source = "\n".join(
        [
            "theorem two (n : Nat) :",
            "    let a := n + 1",
            "    let b := a + 1",
            "    b > n := by",
            "  omega",
        ]
    )

    signature = _signature(source)

    assert signature is not None
    assert signature.endswith("b > n"), signature


def test_a_plain_statement_is_unchanged() -> None:
    signature = _signature("theorem plain (n : Nat) : n = n := rfl")

    assert signature == "(n : Nat) : n = n"


def test_a_multiline_plain_statement_is_unchanged() -> None:
    source = "\n".join(
        [
            "theorem wrapped (n : Nat)",
            "    (h : n > 0) :",
            "    n ≥ 1 := by",
            "  omega",
        ]
    )

    signature = _signature(source)

    assert signature is not None
    assert signature.endswith("n ≥ 1"), signature
    assert "(h : n > 0)" in signature
