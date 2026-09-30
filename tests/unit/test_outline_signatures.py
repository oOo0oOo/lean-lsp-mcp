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


def test_a_trailing_comment_does_not_become_part_of_the_signature() -> None:
    """A `--` comment ends the code on its line, not the declaration."""
    signature = _signature("theorem t (n : Nat) : n = n := rfl  -- TODO: generalise")

    assert signature == "(n : Nat) : n = n"


def test_a_comment_on_a_continuation_line_is_dropped_too() -> None:
    """Stripping happens per physical line, before the lines are joined.

    Joining first would let one line's comment swallow the rest of the
    signature, including the conclusion.
    """
    source = "\n".join(
        [
            "theorem wrapped (n : Nat)  -- the argument",
            "    (h : n > 0) :  -- the hypothesis",
            "    n ≥ 1 := by",
            "  omega",
        ]
    )

    signature = _signature(source)

    assert signature is not None
    assert "--" not in signature, signature
    assert "(h : n > 0)" in signature
    assert signature.endswith("n ≥ 1")


def test_a_doc_comment_opener_is_not_mistaken_for_a_line_comment() -> None:
    """`/--` contains `--`, but the `--` does not follow whitespace."""
    from lean_lsp_mcp.outline_utils import _strip_line_comment

    assert _strip_line_comment("/-- doc -/") == "/-- doc -/"


def test_a_dash_pair_inside_a_string_survives() -> None:
    from lean_lsp_mcp.outline_utils import _strip_line_comment

    assert (
        _strip_line_comment('def t : String := "a -- b"')
        == 'def t : String := "a -- b"'
    )
    assert _strip_line_comment('def t : String := "a" -- b') == 'def t : String := "a"'


def test_a_dash_pair_inside_an_identifier_survives() -> None:
    from lean_lsp_mcp.outline_utils import _strip_line_comment

    assert _strip_line_comment("theorem a--b : True") == "theorem a--b : True"


def _arms(count: int) -> list[str]:
    return [f"  | {i} => let a{i} := {i}; a{i}" for i in range(count)]


def test_match_arms_end_the_signature() -> None:
    """A `let` in an arm spends a `:=` of the body, not of the statement.

    Counting it as a statement binder let the header run on through every
    arm and into the next declaration's.
    """
    source = "\n".join(["def f : Nat → Nat", *_arms(3), "theorem g : True := trivial"])

    declarations = _extract_declarations(source, 0, len(source.splitlines()))

    assert [d["_type"] for d in declarations] == [": Nat → Nat", ": True"]


def test_many_let_arms_stay_linear() -> None:
    import time

    source = "\n".join(
        ["def f : Nat → Nat", *_arms(2000), "theorem g : True := trivial"]
    )

    started = time.perf_counter()
    declarations = _extract_declarations(source, 0, len(source.splitlines()))
    elapsed = time.perf_counter() - started

    assert declarations[0]["_type"] == ": Nat → Nat"
    assert elapsed < 1.0, elapsed


def test_where_ends_the_signature() -> None:
    source = "\n".join(["def origin : Point where", "  x := 0", "  y := 0"])

    assert _signature(source) == ": Point"


def test_an_absolute_value_line_is_not_a_match_arm() -> None:
    source = "\n".join(
        ["theorem abs_bound (x : Int) :", "    |x| ≤ |x| + 1 := by", "  omega"]
    )

    signature = _signature(source)

    assert signature is not None
    assert signature.endswith("|x| ≤ |x| + 1"), signature


def test_a_header_without_body_does_not_borrow_the_next_declaration() -> None:
    source = "\n".join(["def broken : Nat", "theorem g : True := trivial"])

    declarations = _extract_declarations(source, 0, len(source.splitlines()))

    assert declarations[0]["_type"] is None
    assert declarations[1]["_type"] == ": True"
