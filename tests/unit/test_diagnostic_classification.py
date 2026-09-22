from __future__ import annotations

from lean_lsp_mcp.diagnostic_utils import process_diagnostics
from lean_lsp_mcp.models import DiagnosticCategory


def _diag(message: str, severity: int = 1, line: int = 0) -> dict:
    return {
        "severity": severity,
        "message": message,
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 1},
        },
    }


def test_linter_noise_is_separated_from_real_warnings() -> None:
    """Severity alone cannot say which warnings a caller may ignore.

    Lean reports `sorry` usage and unused variables at the same severity as a
    warning that matters, so a caller triaging by severity treats them alike.
    """
    result = process_diagnostics(
        [
            _diag("declaration uses 'sorry'", severity=2, line=0),
            _diag("unused variable `h`\nnote: this linter can be disabled", 2, 1),
            _diag("this looks genuinely suspicious", severity=2, line=2),
        ],
        build_success=True,
    )

    categories = [item.category for item in result.items]
    assert categories == [
        DiagnosticCategory.linter,
        DiagnosticCategory.linter,
        DiagnosticCategory.diagnostic,
    ]


def test_try_this_is_marked_as_a_suggestion() -> None:
    result = process_diagnostics(
        [_diag("Try this: exact foo bar", severity=3)], build_success=True
    )

    assert result.items[0].category is DiagnosticCategory.suggestion


def test_recognised_errors_carry_a_hint() -> None:
    """The message states the symptom; the hint states the remedy."""
    result = process_diagnostics(
        [
            _diag(
                "failed to synthesize instance of type class\n  DecidableEq α",
                line=0,
            ),
            _diag("some error nobody has a hint for", line=1),
        ],
        build_success=True,
    )

    assert result.items[0].hint is not None
    assert "classical" in result.items[0].hint
    assert result.items[1].hint is None


def test_unresolved_binder_hint_only_fires_on_a_metavariable() -> None:
    """A concrete binder type is a different problem from an unresolved one."""
    marker = "invalid binder annotation, type is not a class instance"

    unresolved = process_diagnostics([_diag(f"{marker}\n  ?m.2\n")], build_success=True)
    assert unresolved.items[0].hint is not None
    assert "unresolved" in unresolved.items[0].hint

    concrete = process_diagnostics([_diag(f"{marker}\n  Nat\n")], build_success=True)
    assert concrete.items[0].hint is None


def test_warnings_never_carry_a_hint() -> None:
    """Hints are for what is blocking the caller."""
    result = process_diagnostics(
        [
            _diag(
                "failed to synthesize instance of type class\n  DecidableEq α",
                severity=2,
            )
        ],
        build_success=True,
    )

    assert result.items[0].hint is None


def test_exact_repeats_are_dropped_and_source_order_survives() -> None:
    """A re-elaborated declaration can report the same diagnostic twice.

    Only exact repeats go; a same-message diagnostic at another position is a
    distinct fact about the file and must stay, in the order Lean gave it.
    """
    result = process_diagnostics(
        [
            _diag("type mismatch", line=4),
            _diag("type mismatch", line=4),
            _diag("type mismatch", line=9),
        ],
        build_success=True,
    )

    assert [item.line for item in result.items] == [5, 10]


def test_severity_filter_still_applies_before_dedup() -> None:
    result = process_diagnostics(
        [_diag("a warning", severity=2), _diag("an error", severity=1)],
        build_success=True,
        severity="error",
    )

    assert [item.message for item in result.items] == ["an error"]
