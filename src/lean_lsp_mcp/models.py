"""Pydantic models for MCP tool structured outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field


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
    full_name: str = Field(description="Full qualified name")
    formal_statement: str = Field(description="Lean type signature")
    informal_statement: str = Field(description="Natural language description")


class StateSearchResult(BaseModel):
    name: str = Field(description="Theorem/lemma name")


class PremiseResult(BaseModel):
    name: str = Field(description="Premise name for simp/omega/aesop")


class DiagnosticMessage(BaseModel):
    severity: str = Field(description="error, warning, info, or hint")
    message: str = Field(description="Diagnostic message text")
    line: int = Field(description="Line (1-indexed)")
    column: int = Field(description="Column (1-indexed)")


class GoalState(BaseModel):
    line_context: str = Field(description="Source line where goals were queried")
    goals: str = Field(description="Goal state (before→after if column omitted)")


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


class AttemptResult(BaseModel):
    snippet: str = Field(description="Code snippet that was tried")
    goal_state: Optional[str] = Field(
        None, description="Goal state after applying snippet"
    )
    diagnostics: List[DiagnosticMessage] = Field(
        default_factory=list, description="Diagnostics for this attempt"
    )


class BuildResult(BaseModel):
    success: bool = Field(description="Whether build succeeded")
    output: str = Field(description="Build output")
    errors: List[str] = Field(default_factory=list, description="Build errors if any")


class RunResult(BaseModel):
    success: bool = Field(description="Whether code compiled successfully")
    diagnostics: List[DiagnosticMessage] = Field(
        default_factory=list, description="Compiler diagnostics"
    )


class DeclarationInfo(BaseModel):
    file_path: str = Field(description="Path to declaration file")
    content: str = Field(description="File content")
