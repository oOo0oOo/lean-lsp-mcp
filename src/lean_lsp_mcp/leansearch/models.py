"""Data models for local Lean semantic search.

This module contains the core dataclasses used throughout the leansearch package.
"""

from dataclasses import dataclass, field


@dataclass
class LeanDeclaration:
    """A Lean 4 declaration extracted from source."""

    name: str
    kind: str  # theorem, lemma, def, class, structure, etc.
    module: str  # Module path e.g. Mathlib.Algebra.Group
    signature: str  # Type signature
    docstring: str | None
    file_path: str
    line: int


@dataclass
class IndexStats:
    """Statistics about the indexed project."""

    total_declarations: int = 0
    total_files: int = 0
    declarations_by_kind: dict[str, int] = field(default_factory=dict)
    index_time_seconds: float = 0.0
    project_name: str = ""
    embedding_provider: str = ""
    # Incremental indexing stats
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0


@dataclass
class PremiseEdge:
    """An edge in the premise dependency graph.

    Represents a dependency from one declaration to another.
    """

    target: str  # Name of the declaration being depended on
    is_explicit: bool = False  # Prefixed with * in premises output
    is_simp: bool = False  # Prefixed with s in premises output


@dataclass
class PremiseGraph:
    """A graph of premise dependencies between declarations.

    Built from lean-training-data's `premises` tool output.
    """

    adjacency: dict[str, list[PremiseEdge]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)

    def build_reverse_index(self) -> None:
        """Build reverse lookup: what uses each declaration."""
        self.reverse.clear()
        for name, edges in self.adjacency.items():
            for edge in edges:
                if edge.target not in self.reverse:
                    self.reverse[edge.target] = set()
                self.reverse[edge.target].add(name)
