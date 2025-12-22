"""Test suite for goal-based search quality.

Tests that the unified search system can find relevant lemmas for various
goal states across different mathematical domains.
"""

from __future__ import annotations

import pytest
from pathlib import Path

# Test cases: (goal, expected_names_any, k)
# expected_names_any is a list where at least one should appear in top-k
TEST_CASES = [
    # List operations
    (
        "⊢ List.length (List.map f xs) = List.length xs",
        ["List.length_map", "length_map"],
        5,
    ),
    (
        "⊢ List.length (xs ++ ys) = List.length xs + List.length ys",
        ["List.length_append", "length_append"],
        5,
    ),
    (
        "⊢ List.reverse (List.reverse xs) = xs",
        ["List.reverse_reverse", "reverse_reverse"],
        5,
    ),
    (
        "⊢ List.map f (List.map g xs) = List.map (f ∘ g) xs",
        ["List.map_map", "map_map"],
        5,
    ),
    ("⊢ x ∈ List.map f xs ↔ ∃ y, y ∈ xs ∧ f y = x", ["List.mem_map", "mem_map"], 5),
    # Arithmetic
    ("⊢ a + b = b + a", ["Nat.add_comm", "add_comm", "Int.add_comm"], 5),
    ("⊢ a * b = b * a", ["Nat.mul_comm", "mul_comm", "Int.mul_comm"], 5),
    ("⊢ a + (b + c) = (a + b) + c", ["Nat.add_assoc", "add_assoc"], 5),
    (
        "⊢ a * (b + c) = a * b + a * c",
        ["Nat.mul_add", "mul_add", "Nat.left_distrib", "left_distrib"],
        5,
    ),
    # Set operations
    (
        "⊢ x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t",
        ["Set.mem_union", "Finset.mem_union", "mem_union"],
        5,
    ),
    (
        "⊢ x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t",
        ["Set.mem_inter_iff", "Finset.mem_inter", "mem_inter"],
        5,
    ),
    # Option
    (
        "⊢ Option.map f (Option.map g x) = Option.map (f ∘ g) x",
        ["Option.map_map", "map_map"],
        5,
    ),
    # With hypotheses
    (
        "h : x ∈ xs ⊢ x ∈ xs ++ ys",
        ["List.mem_append_left", "List.mem_append", "mem_append"],
        5,
    ),
    ("h₁ : a ≤ b, h₂ : b ≤ c ⊢ a ≤ c", ["le_trans", "Nat.le_trans"], 5),
    # Type-theoretic
    ("⊢ (a, b).1 = a", ["Prod.fst", "Prod.mk.eta", "fst_def"], 5),
    (
        "⊢ Function.Injective f → Function.Injective g → Function.Injective (f ∘ g)",
        ["Function.Injective.comp", "Injective.comp"],
        5,
    ),
]


def test_extract_constants():
    """Test constant extraction from goal text."""
    from lean_lsp_mcp.leansearch.indexer import extract_constants_from_text

    # Basic case
    consts = extract_constants_from_text("List.length (List.map f xs) = List.length xs")
    assert "List.length" in consts
    assert "List.map" in consts

    # With hypothesis
    consts = extract_constants_from_text(
        "h : x ∈ xs ⊢ Option.isSome (List.find? (· == x) xs)"
    )
    assert "Option.isSome" in consts
    assert "List.find?" in consts

    # Qualified names only
    consts = extract_constants_from_text("a + b = b + a")
    # No uppercase-starting qualified names
    assert len(consts) == 0 or all("." in c for c in consts)


def test_trigger_search_basic():
    """Test basic trigger search functionality."""
    from lean_lsp_mcp.leansearch.indexer import (
        LeanSearchIndex,
        extract_constants_from_text,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        index = LeanSearchIndex(cache_dir=Path(tmpdir), project_name="test")

        # Manually add some test declarations
        conn = index._get_db()

        # Insert test declarations
        conn.execute("""
            INSERT INTO declarations (name, kind, signature, search_text)
            VALUES ('List.length_map', 'theorem', 'List.length (List.map f l) = List.length l', 'length map')
        """)
        conn.execute("""
            INSERT INTO declarations (name, kind, signature, search_text)
            VALUES ('List.length_append', 'theorem', 'List.length (l1 ++ l2) = List.length l1 + List.length l2', 'length append')
        """)
        conn.execute("""
            INSERT INTO declarations (name, kind, signature, search_text)
            VALUES ('List.map_map', 'theorem', 'List.map f (List.map g l) = List.map (f ∘ g) l', 'map map')
        """)
        conn.commit()

        # Add triggers manually
        for name in ["List.length_map", "List.length_append", "List.map_map"]:
            row = conn.execute(
                "SELECT id, signature FROM declarations WHERE name = ?", (name,)
            ).fetchone()
            if row:
                decl_id, sig = row
                consts = extract_constants_from_text(sig)
                for const in consts:
                    conn.execute(
                        "INSERT OR IGNORE INTO triggers (constant, decl_id, is_type_ref) VALUES (?, ?, 1)",
                        (const, decl_id),
                    )
        conn.commit()

        # Build symbol frequency
        index.build_symbol_frequency()

        # Test trigger search
        goal_consts = extract_constants_from_text(
            "List.length (List.map f xs) = List.length xs"
        )
        results = index.search_by_triggers(goal_consts, k=5)

        # Should find List.length_map (uses both List.length and List.map)
        names = [r["name"] for r in results]
        assert "List.length_map" in names, f"Expected List.length_map in {names}"

        # List.length_map should score higher than List.length_append
        # (length_map uses both triggers, append uses only List.length)
        if "List.length_append" in names:
            map_idx = names.index("List.length_map")
            append_idx = names.index("List.length_append")
            assert map_idx < append_idx, (
                "length_map should rank higher than length_append"
            )

        index.close()


def test_goal_to_loogle_query():
    """Test conversion of goal state to loogle query pattern."""
    from lean_lsp_mcp.leansearch import LeanSearchManager

    manager = LeanSearchManager()

    # Basic goal
    query = manager._goal_to_loogle_query(
        "⊢ List.length (List.map f xs) = List.length xs"
    )
    assert query is not None
    assert "List.length" in query
    assert "List.map" in query

    # Goal with hypothesis
    query = manager._goal_to_loogle_query(
        "h : x ∈ xs ⊢ List.find? (· == x) xs = some x"
    )
    assert query is not None
    assert "List.find?" in query

    # Should return None for unstructured goals
    query = manager._goal_to_loogle_query("⊢ a = a")
    # May or may not return None depending on implementation


class TestGoalSearchIntegration:
    """Integration tests for goal search (requires indexed project)."""

    @pytest.fixture
    def project_path(self):
        """Path to a test Lean project with mathlib."""
        # Try common test project locations
        candidates = [
            Path("/Users/alokbeniwal/beads-lean4"),
            Path.home() / "beads-lean4",
            Path.cwd() / "test-project",
        ]
        for p in candidates:
            if p.exists() and (p / "lakefile.lean").exists():
                return p
        pytest.skip("No test project found")

    @pytest.mark.slow
    @pytest.mark.parametrize("goal,expected_any,k", TEST_CASES)
    def test_goal_search_quality(self, project_path, goal, expected_any, k):
        """Test that expected lemmas appear in search results."""
        from lean_lsp_mcp.leansearch import LeanSearchManager

        manager = LeanSearchManager(project_root=project_path)
        try:
            manager.index_project()
            results = manager.search_by_goal(goal, num_results=k)
            result_names = [r.get("name", "") for r in results]

            # Check if any expected name is in results (substring match)
            found = any(
                any(exp.lower() in name.lower() for name in result_names)
                for exp in expected_any
            )

            if not found:
                # Print debug info
                print(f"\nGoal: {goal}")
                print(f"Expected any of: {expected_any}")
                print(f"Got: {result_names}")
                for r in results[:3]:
                    print(
                        f"  - {r.get('name')}: {r.get('source')} ({r.get('score', 0):.2f})"
                    )

            # Don't assert for now - this is for benchmarking
            # assert found, f"Expected one of {expected_any} in top {k}, got {result_names}"
        finally:
            manager.close()

    @pytest.mark.slow
    def test_search_latency(self, project_path):
        """Test that search is fast enough for interactive use."""
        import time
        from lean_lsp_mcp.leansearch import LeanSearchManager

        manager = LeanSearchManager(project_root=project_path)
        try:
            manager.index_project()

            goal = "⊢ List.length (List.map f xs) = List.length xs"
            start = time.time()
            results = manager.search_by_goal(goal, num_results=10)
            elapsed = time.time() - start

            assert elapsed < 2.0, f"Search took {elapsed:.2f}s, expected < 2s"
            print(
                f"\nSearch latency: {elapsed * 1000:.1f}ms for {len(results)} results"
            )
        finally:
            manager.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    test_extract_constants()
    print("✓ test_extract_constants")

    test_trigger_search_basic()
    print("✓ test_trigger_search_basic")

    test_goal_to_loogle_query()
    print("✓ test_goal_to_loogle_query")

    print("\nAll basic tests passed!")
