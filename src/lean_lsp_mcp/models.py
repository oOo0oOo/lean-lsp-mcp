"""Pydantic models for MCP tool structured outputs."""

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer


class ToolOutputModel(BaseModel):
    """Output model that leaves unavailable optional fields off the wire."""

    _omit_empty_fields: ClassVar[frozenset[str]] = frozenset()
    _omit_none_fields: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _omit_empty_optional_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> Any:
        data = handler(self)
        if not isinstance(data, dict):
            return data

        for name in type(self)._omit_none_fields:
            if getattr(self, name) is None:
                field = type(self).model_fields[name]
                key = field.serialization_alias or field.alias or name
                data.pop(key, None)

        for name in type(self)._omit_empty_fields:
            value = getattr(self, name)
            if value is not None and value != "" and value != []:
                continue
            field = type(self).model_fields[name]
            key = field.serialization_alias or field.alias or name
            data.pop(key, None)
        return data


class DiagnosticSeverity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"
    hint = "hint"


class IndexStatus(str, Enum):
    """Whether a local search could consult the language server's symbol index.

    Only ``consulted`` makes an empty result mean the declaration is absent.
    """

    consulted = "consulted"
    # The index is still loading; this search saw source text only.
    warming = "warming"
    # No language server is running for this project. Open a file first.
    unavailable = "unavailable"
    # A server is running but the symbol lookup failed or timed out.
    error = "error"


class LocalSearchResult(ToolOutputModel):
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (theorem, def, class, etc.)")
    file: str = Field(description="Relative file path")


class LeanSearchResult(ToolOutputModel):
    name: str = Field(description="Full qualified name")
    module_name: str = Field(description="Module where declared")
    kind: str | None = Field(None, description="Declaration kind")
    type: str | None = Field(None, description="Type signature")


class LoogleResult(ToolOutputModel):
    name: str = Field(description="Declaration name")
    type: str = Field(description="Type signature")
    module: str = Field(description="Module where declared")


class LeanFinderResult(ToolOutputModel):
    formal_name: str = Field(description="Fully qualified Lean declaration name")
    informal_name: str = Field(
        description="Short natural-language name for the statement"
    )
    kind: str = Field(description="Declaration kind (theorem, def, instance, etc.)")
    type: str = Field(description="Lean type signature / formal statement")
    informal_description: str = Field(
        description="Natural-language description of the statement"
    )
    path: str = Field(
        description="Mathlib module path, dot-separated (e.g. Mathlib.Data.Nat.Basic)"
    )


class PremiseResult(ToolOutputModel):
    name: str = Field(description="Premise name for simp/omega/aesop")


class DiagnosticCategory(str, Enum):
    diagnostic = "diagnostic"
    linter = "linter"
    sorry = "sorry"
    suggestion = "suggestion"


class DiagnosticMessage(ToolOutputModel):
    _omit_empty_fields = frozenset({"lean_tags", "hint"})

    severity: str = Field(description="error, warning, info, or hint")
    message: str = Field(description="Diagnostic message text")
    line: int = Field(description="Line (1-indexed)")
    column: int = Field(description="Column (1-indexed)")
    lean_tags: list[str] | None = Field(
        None,
        description=(
            "Lean-specific tags: 'unsolvedGoals' (proof incomplete here) or "
            "'goalsAccomplished' (proof finished). Machine-readable proof "
            "status - prefer over string-matching the message."
        ),
    )
    category: DiagnosticCategory = Field(
        DiagnosticCategory.diagnostic,
        description=(
            "'linter' is style noise (unused variables and the like) that does "
            "not block compilation; 'sorry' marks an unproved declaration and "
            "is never noise; 'suggestion' carries a `Try this:` replacement; "
            "'diagnostic' is everything else."
        ),
    )
    hint: str | None = Field(
        None,
        description="Guidance for a recognised failure mode, when one applies",
    )


class GoalContextEntry(ToolOutputModel):
    name: str = Field(description="Local hypothesis or variable name")
    type: str = Field(description="Lean type")


class StructuredGoal(ToolOutputModel):
    context: list[GoalContextEntry] = Field(
        default_factory=list, description="Local context entries"
    )
    goal: str | None = Field(None, description="Target goal")
    status: str = Field(description="Goal status: open, complete, or unknown")
    pretty: str = Field(description="Original pretty-printed goal")


GoalOutput = str | StructuredGoal


class GoalState(ToolOutputModel):
    _omit_none_fields = frozenset({"goals", "goals_before", "goals_after", "status"})

    line_context: str = Field(description="Source line where goals were queried")
    goals: list[GoalOutput] | None = Field(
        None, description="Goal list at specified column position"
    )
    goals_before: list[GoalOutput] | None = Field(
        None, description="Goals at line start (when column omitted)"
    )
    goals_after: list[GoalOutput] | None = Field(
        None, description="Goals at line end (when column omitted)"
    )
    status: str | None = Field(
        None,
        description=(
            "Goal status: 'goals' (open goals), 'complete' (no goals left - "
            "proof finished here), 'no_goal_at_position' (position carries no "
            "proof state), or 'still_elaborating' (timeout_s hit - poll again)"
        ),
    )


class CompletionItem(ToolOutputModel):
    _omit_empty_fields = frozenset({"kind", "detail"})

    label: str = Field(description="Completion text to insert")
    kind: str | None = Field(
        None, description="Completion kind (function, variable, etc.)"
    )
    detail: str | None = Field(None, description="Additional detail")


class HoverInfo(ToolOutputModel):
    symbol: str = Field(description="The symbol being hovered")
    info: str = Field(description="Type signature and documentation")
    diagnostics: list[DiagnosticMessage] = Field(
        default_factory=list, description="Diagnostics at this position"
    )


class TermGoalState(ToolOutputModel):
    line_context: str = Field(description="Source line where term goal was queried")
    expected_type: str | None = Field(
        None, description="Expected type at this position"
    )


class OutlineEntry(ToolOutputModel):
    _omit_empty_fields = frozenset({"type_signature", "children"})

    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (Thm, Def, Class, Struct, Ns, Ex)")
    start_line: int = Field(description="Start line (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    type_signature: str | None = Field(None, description="Type signature if available")
    children: list["OutlineEntry"] = Field(
        default_factory=list, description="Nested declarations"
    )


class FileOutline(ToolOutputModel):
    _omit_none_fields = frozenset({"total_declarations"})

    imports: list[str] = Field(default_factory=list, description="Import statements")
    declarations: list[OutlineEntry] = Field(
        default_factory=list, description="Top-level declarations"
    )
    total_declarations: int | None = Field(
        None, description="Total count (set when truncated)"
    )


class AttemptResult(ToolOutputModel):
    _omit_none_fields = frozenset({"proof_status"})

    snippet: str = Field(description="Code snippet that was tried")
    goals: list[str] = Field(
        default_factory=list, description="Goal list after applying snippet"
    )
    diagnostics: list[DiagnosticMessage] = Field(
        default_factory=list, description="Diagnostics for this attempt"
    )
    timed_out: bool = Field(
        False,
        description="True if elaboration timed out (results are partial)",
    )
    proof_status: str | None = Field(
        None,
        description="REPL proof status when available (e.g. 'Completed', 'Incomplete: contains sorry')",
    )


class BuildResult(ToolOutputModel):
    success: bool = Field(description="Whether build succeeded")
    output: str = Field(description="Build output")
    errors: list[str] = Field(default_factory=list, description="Build errors if any")


class RunResult(ToolOutputModel):
    success: bool = Field(description="Whether code compiled successfully")
    timed_out: bool = Field(
        False,
        description="True if elaboration timed out (results are partial)",
    )
    diagnostics: list[DiagnosticMessage] = Field(
        default_factory=list, description="Compiler diagnostics"
    )


class DeclarationInfo(ToolOutputModel):
    _omit_none_fields = frozenset({"start_line", "end_line", "total_lines"})

    file_path: str = Field(description="Path to declaration file")
    content: str = Field(
        description="Declaration source (sliced unless full_file=True)"
    )
    start_line: int | None = Field(
        None, description="First line of the returned slice (1-indexed)"
    )
    end_line: int | None = Field(
        None, description="Last line of the returned slice (1-indexed)"
    )
    declaration_start_line: int | None = Field(
        None, description="Declaration start line (1-indexed)"
    )
    declaration_end_line: int | None = Field(
        None, description="Declaration end line (1-indexed), when known"
    )
    source_declaration: str | None = Field(
        None,
        description=(
            "Source declaration for a generated declaration such as one created "
            "by @[to_additive], when it can be determined reliably"
        ),
    )
    total_lines: int | None = Field(
        None, description="Total lines in the declaration file"
    )


# Wrapper models for list-returning tools
# MCPServer flattens bare lists into separate TextContent blocks, causing serialization issues.
# Wrapping in a model ensures proper JSON serialization.


class DiagnosticsResult(ToolOutputModel):
    """Wrapper for diagnostic messages list with build status."""

    _omit_none_fields = frozenset({"still_elaborating_lines"})
    _omit_empty_fields = frozenset({"failed_dependencies"})

    partial: bool = Field(
        False,
        description=(
            "True when elaboration was still running at timeout: items may be "
            "incomplete and still_elaborating_lines shows the pending region. "
            "Poll again instead of treating this as failure."
        ),
    )
    still_elaborating_lines: list[list[int]] | None = Field(
        None,
        description="Line ranges [start, end] (1-indexed) still being elaborated",
    )

    success: bool = Field(
        True, description="True if the queried file/range has no errors"
    )
    timed_out: bool = Field(
        False,
        description="True if elaboration timed out (results are partial, not a real build failure)",
    )
    items: list[DiagnosticMessage] = Field(
        default_factory=list, description="List of diagnostic messages"
    )
    failed_dependencies: list[str] = Field(
        default_factory=list,
        description="File paths of dependencies that failed to build",
    )


class CompletionsResult(ToolOutputModel):
    """Wrapper for completions list."""

    prefix: str = Field(description="Source prefix completed at the requested position")

    items: list[CompletionItem] = Field(
        default_factory=list, description="List of completion items"
    )


class MultiAttemptResult(ToolOutputModel):
    """Wrapper for multi-attempt results list."""

    items: list[AttemptResult] = Field(
        default_factory=list, description="List of attempt results"
    )


class LocalSearchResults(ToolOutputModel):
    """Wrapper for local search results list."""

    items: list[LocalSearchResult] = Field(
        default_factory=list, description="List of local search results"
    )
    index: IndexStatus = Field(
        IndexStatus.unavailable,
        description=(
            "Whether the language server's symbol index contributed. Only "
            "'consulted' makes an empty result mean the declaration does not "
            "exist. Declarations generated by attributes such as @[to_additive] "
            "exist only in the index, never in source text."
        ),
    )


class LeanSearchResults(ToolOutputModel):
    """Wrapper for LeanSearch results list."""

    items: list[LeanSearchResult] = Field(
        default_factory=list, description="List of LeanSearch results"
    )


class LoogleResults(ToolOutputModel):
    """Wrapper for Loogle results list."""

    items: list[LoogleResult] = Field(
        default_factory=list, description="List of Loogle results"
    )


class LeanFinderResults(ToolOutputModel):
    """Wrapper for Lean Finder results list."""

    items: list[LeanFinderResult] = Field(
        default_factory=list, description="List of Lean Finder results"
    )


class PremiseResults(ToolOutputModel):
    """Wrapper for premise results list."""

    items: list[PremiseResult] = Field(
        default_factory=list, description="List of premise results"
    )


class WidgetsResult(ToolOutputModel):
    """Wrapper for widget instances at a position."""

    widgets: list[dict] = Field(
        default_factory=list, description="Widget instances (id, name, range, props)"
    )


class InteractiveDiagnosticsResult(ToolOutputModel):
    """Wrapper for interactive diagnostics with embedded widgets."""

    diagnostics: list[dict] = Field(
        default_factory=list,
        description="Interactive diagnostic objects with TaggedText messages",
    )


class WidgetSourceResult(ToolOutputModel):
    """Widget JavaScript source for a given hash."""

    source: dict = Field(description="Widget source data including JavaScript module")


class ReferenceLocation(ToolOutputModel):
    """A single reference location."""

    file_path: str = Field(description="Absolute file path")
    line: int = Field(description="Line (1-indexed)")
    column: int = Field(description="Column (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    end_column: int = Field(description="End column (1-indexed)")


class ReferencesResult(ToolOutputModel):
    """Wrapper for find references results."""

    _omit_none_fields = frozenset({"total"})

    items: list[ReferenceLocation] = Field(
        default_factory=list, description="List of reference locations"
    )
    total: int | None = Field(
        None, description="Total matches (> len(items) when truncated by max_results)"
    )


class LineProfile(ToolOutputModel):
    """Timing for a single source line."""

    line: int = Field(description="Source line number (1-indexed)")
    ms: float = Field(description="Time in milliseconds")
    text: str = Field(description="Source line content (truncated)")


class ProofProfileResult(ToolOutputModel):
    """Profiling result for a theorem."""

    ms: float = Field(description="Total elaboration time in ms")
    lines: list[LineProfile] = Field(
        default_factory=list, description="Time per source line (>1% of total)"
    )
    categories: dict[str, float] = Field(
        default_factory=dict, description="Cumulative time by category in ms"
    )


class CodeActionEdit(ToolOutputModel):
    new_text: str = Field(description="Replacement text")
    start_line: int = Field(description="Start line (1-indexed)")
    start_column: int = Field(description="Start column (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    end_column: int = Field(description="End column (1-indexed)")


class CodeAction(ToolOutputModel):
    title: str = Field(
        description="Code action title (e.g. 'Try this: simp only [...])"
    )
    is_preferred: bool = Field(description="Whether this is the preferred action")
    edits: list[CodeActionEdit] = Field(
        default_factory=list, description="Text edits to apply"
    )


class CodeActionsResult(ToolOutputModel):
    """Wrapper for code actions at a position."""

    actions: list[CodeAction] = Field(
        default_factory=list, description="List of available code actions"
    )


class SourceWarning(ToolOutputModel):
    line: int = Field(description="Line number (1-indexed)")
    pattern: str = Field(description="Matched pattern text")


class AxiomTrust(str, Enum):
    standard = "standard"
    # `sorryAx`: the theorem is not proved.
    incomplete = "incomplete"
    # `native_decide`: rests on the compiler and runtime, not the kernel.
    native = "native"
    # Project or other axioms beyond the standard three.
    custom = "custom"


class VerifyResult(ToolOutputModel):
    axioms: list[str] = Field(
        default_factory=list,
        description="Axioms used. Standard 3: propext, Classical.choice, Quot.sound",
    )
    trust: AxiomTrust = Field(
        AxiomTrust.standard,
        description=(
            "Worst case in `axioms`. 'standard' is the ordinary Mathlib basis; "
            "'incomplete' means sorryAx, so it is not a proof; 'native' means "
            "native_decide, trusting the compiler rather than the kernel; "
            "'custom' means other axioms are in play."
        ),
    )
    non_standard_axioms: list[str] = Field(
        default_factory=list,
        description="Axioms outside the standard 3 — the reason for `trust`",
    )
    warnings: list[SourceWarning] = Field(
        default_factory=list,
        description="Suspicious source patterns (if enabled)",
    )


class HypothesisStatus(str, Enum):
    load_bearing = "load-bearing"
    removable = "removable"
    error = "error"


class HypothesisVerdict(ToolOutputModel):
    _omit_empty_fields = frozenset({"detail"})

    binder: str = Field(description="Lean source of the binder, e.g. '(h : P)'")
    status: HypothesisStatus = Field(
        description=(
            "load-bearing: removing this binder caused new errors. "
            "removable: the file still elaborated cleanly without it. "
            "error: probing this binder failed (e.g. LSP timeout)."
        )
    )
    breaks: list[DiagnosticMessage] = Field(
        default_factory=list,
        description=(
            "New errors caused by removing this binder (empty when removable). "
            "Each entry has line/column/message — useful for finding where in "
            "the proof body the dropped hypothesis was used."
        ),
    )
    detail: str = Field(
        default="",
        description="Free-text status detail (e.g. timeout reason)",
    )


class MinimalHypothesesResult(ToolOutputModel):
    theorem_name: str = Field(description="Theorem analyzed")
    file: str = Field(description="Relative file path")
    verdicts: list[HypothesisVerdict] = Field(
        default_factory=list,
        description="One verdict per explicit (h : T) binder, in source order",
    )
    skipped_implicit: int = Field(
        default=0,
        description="Count of implicit {x : α} / instance [inst] binders not probed",
    )
