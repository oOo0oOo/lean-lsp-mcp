from __future__ import annotations

import pytest

from lean_lsp_mcp import config
from lean_lsp_mcp.diagnostic_utils import process_diagnostics
from lean_lsp_mcp.utils import bound_output


def _diag(message: str) -> dict:
    return {
        "severity": 1,
        "message": message,
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
    }


def test_short_text_is_returned_whole() -> None:
    assert bound_output("⊢ True") == "⊢ True"


def test_the_middle_goes_and_the_target_survives(monkeypatch) -> None:
    """A Lean goal states its target last, after the local context.

    Cutting the tail would drop the one part the caller came for, so the
    elision takes the middle and keeps both ends.
    """
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "300")

    goal = "h : " + ("A" * 5000) + "\n⊢ TheTargetIWanted"
    bounded = bound_output(goal)

    assert bounded.startswith("h : AAA")
    assert bounded.endswith("⊢ TheTargetIWanted")
    assert "characters elided" in bounded
    assert len(bounded) < len(goal)


def test_the_elision_says_how_much_went(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "100")

    bounded = bound_output("X" * 1100)

    assert "1000 characters elided" in bounded


def test_zero_disables_the_budget(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "0")

    text = "Y" * 50_000
    assert bound_output(text) == text


@pytest.mark.parametrize("raw", ["not-a-number", "-5"])
def test_a_bad_budget_falls_back_instead_of_raising(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, raw)

    assert config.max_output_chars() >= 0


def test_diagnostics_are_classified_on_the_full_text(monkeypatch) -> None:
    """The marker that identifies a failure mode may sit in the elided middle.

    Classifying the trimmed text would lose the hint precisely on the largest
    diagnostics, which are the ones a caller most needs help with.
    """
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "200")

    message = (
        "failed to synthesize instance of type class\n  DecidableEq α\n"
        + ("padding line\n" * 400)
        + "tail"
    )
    result = process_diagnostics([_diag(message)], build_success=True)

    item = result.items[0]
    assert item.hint is not None
    assert "characters elided" in item.message
    assert len(item.message) < len(message)


def test_distinct_diagnostics_survive_identical_truncation(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "200")
    first = "A" * 500 + "first error" + "Z" * 500
    second = "A" * 500 + "other error" + "Z" * 500
    assert bound_output(first) == bound_output(second)

    result = process_diagnostics(
        [_diag(first), _diag(second), _diag(first)], build_success=False
    )
    assert len(result.items) == 2


def test_build_failures_are_extracted_before_truncation(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_OUTPUT_CHARS_ENV, "200")
    message = (
        "lake setup-file failed\n"
        + "padding\n" * 100
        + "error: Dependency.lean:12:3: unknown identifier\n"
        + "padding\n" * 100
    )
    result = process_diagnostics([_diag(message)], build_success=False)
    assert result.failed_dependencies == ["Dependency.lean"]
