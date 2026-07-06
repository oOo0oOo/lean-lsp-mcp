"""Pydantic models for MCP tool structured outputs."""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class DiagnosticSeverity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"
    hint = "hint"


class LocalSearchResult(BaseModel):
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (theorem, def, class, etc.)")
    file: str = Field(description="Relative file path")


class LeanSearchResult(BaseModel):
    name: str = Field(description="Full qualified name")
    module_name: str = Field(description="Module where declared")
    kind: Optional[str] = Field(None, description="Declaration kind")
    type: Optional[str] = Field(None, description="Type signature")


class LoogleResult(BaseModel):
    name: str = Field(description="Declaration name")
    type: str = Field(description="Type signature")
    module: str = Field(description="Module where declared")


class LeanFinderResult(BaseModel):
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


class StateSearchResult(BaseModel):
    name: str = Field(description="Theorem/lemma name")


class PremiseResult(BaseModel):
    name: str = Field(description="Premise name for simp/omega/aesop")


class DiagnosticMessage(BaseModel):
    severity: str = Field(description="error, warning, info, or hint")
    message: str = Field(description="Diagnostic message text")
    line: int = Field(description="Line (1-indexed)")
    column: int = Field(description="Column (1-indexed)")
    lean_tags: Optional[List[str]] = Field(
        None,
        description=(
            "Lean-specific tags: 'unsolvedGoals' (proof incomplete here) or "
            "'goalsAccomplished' (proof finished). Machine-readable proof "
            "status - prefer over string-matching the message."
        ),
    )


class GoalContextEntry(BaseModel):
    name: str = Field(description="Local hypothesis or variable name")
    type: str = Field(description="Lean type")


class StructuredGoal(BaseModel):
    context: List[GoalContextEntry] = Field(
        default_factory=list, description="Local context entries"
    )
    goal: Optional[str] = Field(None, description="Target goal")
    status: str = Field(description="Goal status: open, complete, or unknown")
    pretty: str = Field(description="Original pretty-printed goal")


GoalOutput = Union[str, StructuredGoal]


class GoalState(BaseModel):
    line_context: str = Field(description="Source line where goals were queried")
    goals: Optional[List[GoalOutput]] = Field(
        None, description="Goal list at specified column position"
    )
    goals_before: Optional[List[GoalOutput]] = Field(
        None, description="Goals at line start (when column omitted)"
    )
    goals_after: Optional[List[GoalOutput]] = Field(
        None, description="Goals at line end (when column omitted)"
    )
    status: Optional[str] = Field(
        None,
        description=(
            "Goal status: 'goals' (open goals), 'complete' (no goals left - "
            "proof finished here), 'no_goal_at_position' (position carries no "
            "proof state), or 'still_elaborating' (timeout_s hit - poll again)"
        ),
    )


class CompletionItem(BaseModel):
    label: str = Field(description="Completion text to insert")
    kind: Optional[str] = Field(
        None, description="Completion kind (function, variable, etc.)"
    )
    detail: Optional[str] = Field(None, description="Additional detail")


class HoverInfo(BaseModel):
    symbol: str = Field(description="The symbol being hovered")
    info: str = Field(description="Type signature and documentation")
    diagnostics: List[DiagnosticMessage] = Field(
        default_factory=list, description="Diagnostics at this position"
    )


class TermGoalState(BaseModel):
    line_context: str = Field(description="Source line where term goal was queried")
    expected_type: Optional[str] = Field(
        None, description="Expected type at this position"
    )


class OutlineEntry(BaseModel):
    name: str = Field(description="Declaration name")
    kind: str = Field(description="Declaration kind (Thm, Def, Class, Struct, Ns, Ex)")
    start_line: int = Field(description="Start line (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    type_signature: Optional[str] = Field(
        None, description="Type signature if available"
    )
    children: List["OutlineEntry"] = Field(
        default_factory=list, description="Nested declarations"
    )


class FileOutline(BaseModel):
    imports: List[str] = Field(default_factory=list, description="Import statements")
    declarations: List[OutlineEntry] = Field(
        default_factory=list, description="Top-level declarations"
    )
    total_declarations: Optional[int] = Field(
        None, description="Total count (set when truncated)"
    )


class AttemptResult(BaseModel):
    snippet: str = Field(description="Code snippet that was tried")
    goals: List[str] = Field(
        default_factory=list, description="Goal list after applying snippet"
    )
    diagnostics: List[DiagnosticMessage] = Field(
        default_factory=list, description="Diagnostics for this attempt"
    )
    timed_out: bool = Field(
        False,
        description="True if elaboration timed out (results are partial)",
    )
    proof_status: Optional[str] = Field(
        None,
        description="REPL proof status when available (e.g. 'Completed', 'Incomplete: contains sorry')",
    )


class BuildResult(BaseModel):
    success: bool = Field(description="Whether build succeeded")
    output: str = Field(description="Build output")
    errors: List[str] = Field(default_factory=list, description="Build errors if any")


class RunResult(BaseModel):
    success: bool = Field(description="Whether code compiled successfully")
    timed_out: bool = Field(
        False,
        description="True if elaboration timed out (results are partial)",
    )
    diagnostics: List[DiagnosticMessage] = Field(
        default_factory=list, description="Compiler diagnostics"
    )


class DeclarationInfo(BaseModel):
    file_path: str = Field(description="Path to declaration file")
    content: str = Field(
        description="Declaration source (sliced unless full_file=True)"
    )
    start_line: Optional[int] = Field(
        None, description="First line of the returned slice (1-indexed)"
    )
    end_line: Optional[int] = Field(
        None, description="Last line of the returned slice (1-indexed)"
    )
    total_lines: Optional[int] = Field(
        None, description="Total lines in the declaration file"
    )


# Wrapper models for list-returning tools
# FastMCP flattens bare lists into separate TextContent blocks, causing serialization issues.
# Wrapping in a model ensures proper JSON serialization.


class DiagnosticsResult(BaseModel):
    """Wrapper for diagnostic messages list with build status."""

    partial: bool = Field(
        False,
        description=(
            "True when elaboration was still running at timeout: items may be "
            "incomplete and still_elaborating_lines shows the pending region. "
            "Poll again instead of treating this as failure."
        ),
    )
    still_elaborating_lines: Optional[List[List[int]]] = Field(
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
    items: List[DiagnosticMessage] = Field(
        default_factory=list, description="List of diagnostic messages"
    )
    failed_dependencies: List[str] = Field(
        default_factory=list,
        description="File paths of dependencies that failed to build",
    )


class CompletionsResult(BaseModel):
    """Wrapper for completions list."""

    items: List[CompletionItem] = Field(
        default_factory=list, description="List of completion items"
    )


class MultiAttemptResult(BaseModel):
    """Wrapper for multi-attempt results list."""

    items: List[AttemptResult] = Field(
        default_factory=list, description="List of attempt results"
    )


class LocalSearchResults(BaseModel):
    """Wrapper for local search results list."""

    items: List[LocalSearchResult] = Field(
        default_factory=list, description="List of local search results"
    )


class LeanSearchResults(BaseModel):
    """Wrapper for LeanSearch results list."""

    items: List[LeanSearchResult] = Field(
        default_factory=list, description="List of LeanSearch results"
    )


class LoogleResults(BaseModel):
    """Wrapper for Loogle results list."""

    items: List[LoogleResult] = Field(
        default_factory=list, description="List of Loogle results"
    )


class LeanFinderResults(BaseModel):
    """Wrapper for Lean Finder results list."""

    items: List[LeanFinderResult] = Field(
        default_factory=list, description="List of Lean Finder results"
    )


class StateSearchResults(BaseModel):
    """Wrapper for state search results list."""

    items: List[StateSearchResult] = Field(
        default_factory=list, description="List of state search results"
    )


class PremiseResults(BaseModel):
    """Wrapper for premise results list."""

    items: List[PremiseResult] = Field(
        default_factory=list, description="List of premise results"
    )


class WidgetsResult(BaseModel):
    """Wrapper for widget instances at a position."""

    widgets: List[dict] = Field(
        default_factory=list, description="Widget instances (id, name, range, props)"
    )


class InteractiveDiagnosticsResult(BaseModel):
    """Wrapper for interactive diagnostics with embedded widgets."""

    diagnostics: List[dict] = Field(
        default_factory=list,
        description="Interactive diagnostic objects with TaggedText messages",
    )


class WidgetSourceResult(BaseModel):
    """Widget JavaScript source for a given hash."""

    source: dict = Field(description="Widget source data including JavaScript module")


class ReferenceLocation(BaseModel):
    """A single reference location."""

    file_path: str = Field(description="Absolute file path")
    line: int = Field(description="Line (1-indexed)")
    column: int = Field(description="Column (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    end_column: int = Field(description="End column (1-indexed)")


class ReferencesResult(BaseModel):
    """Wrapper for find references results."""

    items: List[ReferenceLocation] = Field(
        default_factory=list, description="List of reference locations"
    )
    total: Optional[int] = Field(
        None, description="Total matches (> len(items) when truncated by max_results)"
    )


class LineProfile(BaseModel):
    """Timing for a single source line."""

    line: int = Field(description="Source line number (1-indexed)")
    ms: float = Field(description="Time in milliseconds")
    text: str = Field(description="Source line content (truncated)")


class ProofProfileResult(BaseModel):
    """Profiling result for a theorem."""

    ms: float = Field(description="Total elaboration time in ms")
    lines: List[LineProfile] = Field(
        default_factory=list, description="Time per source line (>1% of total)"
    )
    categories: dict[str, float] = Field(
        default_factory=dict, description="Cumulative time by category in ms"
    )


class CodeActionEdit(BaseModel):
    new_text: str = Field(description="Replacement text")
    start_line: int = Field(description="Start line (1-indexed)")
    start_column: int = Field(description="Start column (1-indexed)")
    end_line: int = Field(description="End line (1-indexed)")
    end_column: int = Field(description="End column (1-indexed)")


class CodeAction(BaseModel):
    title: str = Field(
        description="Code action title (e.g. 'Try this: simp only [...])"
    )
    is_preferred: bool = Field(description="Whether this is the preferred action")
    edits: List[CodeActionEdit] = Field(
        default_factory=list, description="Text edits to apply"
    )


class CodeActionsResult(BaseModel):
    """Wrapper for code actions at a position."""

    actions: List[CodeAction] = Field(
        default_factory=list, description="List of available code actions"
    )


class SourceWarning(BaseModel):
    line: int = Field(description="Line number (1-indexed)")
    pattern: str = Field(description="Matched pattern text")


class VerifyResult(BaseModel):
    axioms: List[str] = Field(
        default_factory=list,
        description="Axioms used. Standard 3: propext, Classical.choice, Quot.sound",
    )
    warnings: List[SourceWarning] = Field(
        default_factory=list,
        description="Suspicious source patterns (if enabled)",
    )


class HypothesisStatus(str, Enum):
    load_bearing = "load-bearing"
    removable = "removable"
    error = "error"


class HypothesisVerdict(BaseModel):
    binder: str = Field(description="Lean source of the binder, e.g. '(h : P)'")
    status: HypothesisStatus = Field(
        description=(
            "load-bearing: removing this binder caused new errors. "
            "removable: the file still elaborated cleanly without it. "
            "error: probing this binder failed (e.g. LSP timeout)."
        )
    )
    breaks: List[DiagnosticMessage] = Field(
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


class MinimalHypothesesResult(BaseModel):
    theorem_name: str = Field(description="Theorem analyzed")
    file: str = Field(description="Relative file path")
    verdicts: List[HypothesisVerdict] = Field(
        default_factory=list,
        description="One verdict per explicit (h : T) binder, in source order",
    )
    skipped_implicit: int = Field(
        default=0,
        description="Count of implicit {x : α} / instance [inst] binders not probed",
    )
