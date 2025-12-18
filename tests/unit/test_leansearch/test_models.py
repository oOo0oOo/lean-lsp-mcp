"""Tests for leansearch.models module."""

import pytest
from lean_lsp_mcp.leansearch.models import (
    LeanDeclaration,
    IndexStats,
    PremiseEdge,
    PremiseGraph,
)


class TestLeanDeclaration:
    def test_create_declaration(self):
        decl = LeanDeclaration(
            name="Nat.add_comm",
            kind="theorem",
            module="Mathlib.Algebra.Group",
            signature="∀ (n m : Nat), n + m = m + n",
            docstring="Addition is commutative",
            file_path="/path/to/file.lean",
            line=42,
        )
        assert decl.name == "Nat.add_comm"
        assert decl.kind == "theorem"
        assert decl.module == "Mathlib.Algebra.Group"
        assert decl.line == 42

    def test_declaration_with_none_docstring(self):
        decl = LeanDeclaration(
            name="List.map",
            kind="def",
            module="Std.Data.List",
            signature="(f : α → β) → List α → List β",
            docstring=None,
            file_path="/path/to/file.lean",
            line=10,
        )
        assert decl.docstring is None


class TestIndexStats:
    def test_default_values(self):
        stats = IndexStats()
        assert stats.total_declarations == 0
        assert stats.total_files == 0
        assert stats.declarations_by_kind == {}
        assert stats.index_time_seconds == 0.0

    def test_with_values(self):
        stats = IndexStats(
            total_declarations=1000,
            total_files=50,
            declarations_by_kind={"theorem": 500, "def": 300, "lemma": 200},
            index_time_seconds=5.5,
            project_name="test_project",
            files_added=10,
            files_updated=5,
            files_unchanged=35,
        )
        assert stats.total_declarations == 1000
        assert stats.declarations_by_kind["theorem"] == 500
        assert stats.files_added == 10


class TestPremiseGraph:
    def test_empty_graph(self):
        graph = PremiseGraph()
        assert graph.adjacency == {}
        assert graph.reverse == {}

    def test_build_reverse_index(self):
        graph = PremiseGraph()
        graph.adjacency = {
            "theorem_a": [
                PremiseEdge(target="lemma_x", is_explicit=True),
                PremiseEdge(target="lemma_y", is_explicit=False),
            ],
            "theorem_b": [
                PremiseEdge(target="lemma_x", is_explicit=True),
            ],
        }
        graph.build_reverse_index()

        # lemma_x should be used by both theorem_a and theorem_b
        assert "lemma_x" in graph.reverse
        assert "theorem_a" in graph.reverse["lemma_x"]
        assert "theorem_b" in graph.reverse["lemma_x"]

        # lemma_y should only be used by theorem_a
        assert "lemma_y" in graph.reverse
        assert "theorem_a" in graph.reverse["lemma_y"]
        assert "theorem_b" not in graph.reverse["lemma_y"]


class TestPremiseEdge:
    def test_explicit_edge(self):
        edge = PremiseEdge(target="Nat.add", is_explicit=True, is_simp=False)
        assert edge.target == "Nat.add"
        assert edge.is_explicit is True
        assert edge.is_simp is False

    def test_simp_edge(self):
        edge = PremiseEdge(target="Nat.succ_add", is_explicit=False, is_simp=True)
        assert edge.is_simp is True
        assert edge.is_explicit is False

    def test_default_values(self):
        edge = PremiseEdge(target="some_lemma")
        assert edge.is_explicit is False
        assert edge.is_simp is False
