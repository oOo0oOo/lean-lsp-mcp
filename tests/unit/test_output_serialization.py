"""Output models omit unavailable optional fields without changing schemas."""

from jsonschema import Draft202012Validator

from lean_lsp_mcp.models import (
    DiagnosticMessage,
    DiagnosticsResult,
    FileOutline,
    GoalState,
    OutlineEntry,
    StructuredGoal,
    TermGoalState,
)


def test_diagnostics_omit_unavailable_optional_fields_and_validate_schema():
    result = DiagnosticsResult(
        success=False,
        items=[
            DiagnosticMessage(
                severity="error",
                message="type mismatch",
                line=3,
                column=16,
            )
        ],
    )

    payload = result.model_dump(mode="json")

    assert "still_elaborating_lines" not in payload
    assert "failed_dependencies" not in payload
    assert "lean_tags" not in payload["items"][0]
    assert "hint" not in payload["items"][0]
    Draft202012Validator(DiagnosticsResult.model_json_schema()).validate(payload)


def test_goal_state_omits_inapplicable_fields_but_keeps_empty_goal_lists():
    result = GoalState(
        line_context="  trivial",
        goals_before=["⊢ True"],
        goals_after=[],
    )

    payload = result.model_dump(mode="json")

    assert "goals" not in payload
    assert "status" not in payload
    assert payload["goals_before"] == ["⊢ True"]
    assert payload["goals_after"] == []
    Draft202012Validator(GoalState.model_json_schema()).validate(payload)


def test_outline_omits_empty_optional_fields_and_validates_schema():
    result = FileOutline(
        declarations=[OutlineEntry(name="plain", kind="Thm", start_line=1, end_line=1)]
    )

    payload = result.model_dump(mode="json")

    assert "total_declarations" not in payload
    assert "type_signature" not in payload["declarations"][0]
    assert "children" not in payload["declarations"][0]
    Draft202012Validator(FileOutline.model_json_schema()).validate(payload)


def test_fields_with_meaningful_null_values_are_preserved():
    term_goal = TermGoalState(line_context="x", expected_type=None).model_dump(
        mode="json"
    )
    complete_goal = StructuredGoal(
        status="complete", pretty="no goals", goal=None
    ).model_dump(mode="json")

    assert term_goal["expected_type"] is None
    assert complete_goal["goal"] is None
