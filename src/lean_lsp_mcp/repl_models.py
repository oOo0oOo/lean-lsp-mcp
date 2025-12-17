"""Pydantic models for Lean REPL JSON protocol.

These models match the leanprover-community/repl JSON API for:
- Command execution with environment tracking
- Tactic mode for step-by-step proof development
- File loading with tactic extraction
- Environment pickling/unpickling
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Request Models (sent to REPL subprocess)
# ============================================================================


class ReplCommandRequest(BaseModel):
    """Execute a Lean command (def, theorem, #check, etc.) in an environment."""

    cmd: str = Field(description="Lean command to execute")
    env: Optional[int] = Field(
        None, description="Environment ID to use (omit for fresh environment)"
    )


class ReplTacticRequest(BaseModel):
    """Apply a tactic in proof state (tactic mode)."""

    tactic: str = Field(description="Tactic to apply")
    proofState: int = Field(description="Proof state ID from sorry response")


class ReplFileRequest(BaseModel):
    """Load and process a Lean file."""

    path: str = Field(description="Path to Lean file (relative to project)")
    allTactics: bool = Field(
        False, description="Include all tactics in response for analysis"
    )


class ReplPickleEnvRequest(BaseModel):
    """Serialize environment to .olean file."""

    pickleTo: str = Field(description="Path to write .olean file")
    env: int = Field(description="Environment ID to pickle")


class ReplPickleProofStateRequest(BaseModel):
    """Serialize proof state to .olean file."""

    pickleTo: str = Field(description="Path to write .olean file")
    proofState: int = Field(description="Proof state ID to pickle")


class ReplUnpickleEnvRequest(BaseModel):
    """Restore environment from .olean file."""

    unpickleEnvFrom: str = Field(description="Path to .olean file")


class ReplUnpickleProofStateRequest(BaseModel):
    """Restore proof state from .olean file."""

    unpickleProofStateFrom: str = Field(description="Path to .olean file")


# Union of all request types for type checking
ReplRequest = Union[
    ReplCommandRequest,
    ReplTacticRequest,
    ReplFileRequest,
    ReplPickleEnvRequest,
    ReplPickleProofStateRequest,
    ReplUnpickleEnvRequest,
    ReplUnpickleProofStateRequest,
]


# ============================================================================
# Response Models (received from REPL subprocess)
# ============================================================================


class ReplPosition(BaseModel):
    """Position in source code."""

    line: int = Field(description="Line number (1-indexed)")
    column: int = Field(description="Column number (1-indexed)")


class ReplSorry(BaseModel):
    """Information about a sorry placeholder in the code."""

    pos: ReplPosition = Field(description="Position of the sorry")
    goal: str = Field(description="Goal to prove at this sorry")
    proofState: int = Field(description="Proof state ID for tactic mode")
    endPos: Optional[ReplPosition] = Field(None, description="End position of sorry")


class ReplMessage(BaseModel):
    """Compiler diagnostic message."""

    severity: Literal["error", "warning", "info", "trace"] = Field(
        description="Message severity level"
    )
    pos: ReplPosition = Field(description="Position of the message")
    endPos: Optional[ReplPosition] = Field(None, description="End position")
    data: str = Field(description="Message content")


class ReplTacticInfo(BaseModel):
    """Information about a tactic (from file mode with allTactics=true)."""

    tactic: str = Field(description="The tactic that was applied")
    goals: List[str] = Field(default_factory=list, description="Goals after tactic")
    pos: ReplPosition = Field(description="Tactic position")
    endPos: Optional[ReplPosition] = Field(None, description="End position")


class ReplCommandResponse(BaseModel):
    """Response from command or file execution."""

    env: Optional[int] = Field(None, description="New environment ID")
    sorries: List[ReplSorry] = Field(
        default_factory=list, description="Sorry placeholders found"
    )
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Compiler messages"
    )
    tactics: List[ReplTacticInfo] = Field(
        default_factory=list, description="Tactics found (file mode with allTactics)"
    )


class ReplTacticResponse(BaseModel):
    """Response from tactic application."""

    proofState: Optional[int] = Field(
        None, description="New proof state ID (None if proof complete)"
    )
    goals: List[str] = Field(
        default_factory=list, description="Remaining goals after tactic"
    )
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Messages from tactic execution"
    )
    traces: List[str] = Field(default_factory=list, description="Trace output")
    proofStatus: Optional[str] = Field(
        None, description="Proof completion status"
    )


class ReplErrorResponse(BaseModel):
    """Error response from REPL."""

    error: str = Field(description="Error message")


# ============================================================================
# MCP Tool Result Wrappers
# ============================================================================


class ReplCmdResult(BaseModel):
    """Result of lean_repl_cmd tool."""

    env: Optional[int] = Field(None, description="New environment ID")
    sorries: List[ReplSorry] = Field(
        default_factory=list, description="Sorry placeholders with proof state IDs"
    )
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Compiler messages"
    )
    success: bool = Field(description="True if no errors")


class ReplTacticResult(BaseModel):
    """Result of lean_repl_tactic tool."""

    proofState: Optional[int] = Field(
        None, description="New proof state ID (None if proof complete)"
    )
    goals: List[str] = Field(
        default_factory=list, description="Remaining goals after tactic"
    )
    success: bool = Field(description="True if tactic succeeded")
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Messages from tactic execution"
    )


class ReplFileResult(BaseModel):
    """Result of lean_repl_file tool."""

    env: Optional[int] = Field(None, description="Environment after loading file")
    sorries: List[ReplSorry] = Field(
        default_factory=list, description="Sorry placeholders found"
    )
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Compiler messages"
    )
    tactics: List[ReplTacticInfo] = Field(
        default_factory=list, description="Tactics found (if allTactics=true)"
    )
    success: bool = Field(description="True if no errors")


class ReplPickleResult(BaseModel):
    """Result of pickling operation."""

    success: bool = Field(description="True if pickle succeeded")
    path: str = Field(description="Path where pickle was written")
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Messages from operation"
    )


class ReplUnpickleResult(BaseModel):
    """Result of unpickling operation."""

    env: Optional[int] = Field(None, description="Restored environment ID")
    proofState: Optional[int] = Field(None, description="Restored proof state ID")
    success: bool = Field(description="True if unpickle succeeded")
    messages: List[ReplMessage] = Field(
        default_factory=list, description="Messages from operation"
    )


# ============================================================================
# Session Management Models
# ============================================================================


class ReplSessionInfo(BaseModel):
    """Information about a REPL session."""

    session_id: str = Field(description="Unique session identifier")
    project_path: str = Field(description="Path to Lean project")
    current_env: Optional[int] = Field(
        None, description="Most recent environment ID in session"
    )
    proof_states: List[int] = Field(
        default_factory=list, description="Active proof state IDs"
    )
    created_at: float = Field(description="Session creation timestamp")


class ReplSessionsResult(BaseModel):
    """List of active REPL sessions."""

    sessions: List[ReplSessionInfo] = Field(
        default_factory=list, description="Active sessions"
    )


class ReplSessionCreateResult(BaseModel):
    """Result of creating a new session."""

    session_id: str = Field(description="New session ID")
    project_path: str = Field(description="Associated project path")


class ReplSessionDeleteResult(BaseModel):
    """Result of deleting a session."""

    success: bool = Field(description="True if session was deleted")
    session_id: str = Field(description="Deleted session ID")


class ReplMultiTacticResult(BaseModel):
    """Result of trying multiple tactics."""

    results: List[ReplTacticResult] = Field(
        default_factory=list, description="Result for each tactic tried"
    )
