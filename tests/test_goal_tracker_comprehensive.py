"""Comprehensive comparison: lean_verify vs lean_goal_tracker vs ground truth.

Tests two categories:
  A) Single-file tests (GoalTrackerTest.lean) — all declarations in one file
  B) Cross-file tests (GoalTrackerBase.lean + GoalTrackerImport.lean) —
     sorry dependencies across file boundaries via imports

Ground truth from `#print axioms` via `lake env lean`:

  gt_clean              -> [propext]                      (no sorry)
  gt_direct_sorry       -> [sorryAx]                      (direct sorry)
  gt_helper             -> no axioms                      (clean def)
  gt_uses_clean_helper  -> no axioms                      (clean thm using clean def)
  gt_sorry_def          -> [sorryAx]                      (sorry in def)
  gt_transitive_sorry   -> [sorryAx]                      (sorry def + sorry proof)
  gt_level2_sorry       -> [sorryAx]                      (sorry def, depth 2)
  gt_level1             -> [sorryAx]                      (depends on sorry def)
  gt_chain_sorry        -> [sorryAx]                      (2-level chain)
  gt_decidable          -> no axioms                      (clean)
  gt_noncomputable      -> [Classical.choice]             (no sorry, has axiom)
  gt_uses_noncomputable -> [Classical.choice]             (transitive axiom, no sorry)
  gt_term_sorry         -> [sorryAx]                      (term-mode sorry)
  gt_uses_term_sorry    -> [sorryAx]                      (transitive term sorry)
  gt_sorry_a            -> [sorryAx]                      (sorry)
  gt_sorry_b            -> [sorryAx]                      (sorry)
  gt_multi_sorry        -> [sorryAx]                      (multiple sorry sources)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


def sorry_names(data: dict) -> list[str]:
    """Extract sorry leaf names from goal_tracker result."""
    return [s["name"] for s in data["sorry_declarations"]]


# Ground truth: does #print axioms report sorryAx?
GROUND_TRUTH = {
    # (has_sorry_in_axioms, expected_sorry_decls_subset, description)
    "gt_clean": (False, [], "clean theorem, standard axioms only"),
    "gt_direct_sorry": (True, ["gt_direct_sorry"], "direct sorry in proof"),
    "gt_helper": (False, [], "clean def, no axioms"),
    "gt_uses_clean_helper": (False, [], "clean theorem using clean def"),
    "gt_sorry_def": (True, ["gt_sorry_def"], "sorry in def body"),
    "gt_transitive_sorry": (
        True,
        ["gt_transitive_sorry", "gt_sorry_def"],
        "sorry def + sorry proof",
    ),
    "gt_level2_sorry": (True, ["gt_level2_sorry"], "sorry def at depth 2"),
    "gt_level1": (True, ["gt_level2_sorry"], "depends on sorry def transitively"),
    "gt_chain_sorry": (True, ["gt_level2_sorry"], "2-level chain to sorry"),
    "gt_decidable": (False, [], "decidable, no sorry"),
    "gt_noncomputable": (False, [], "Classical.choice axiom, no sorry"),
    "gt_uses_noncomputable": (False, [], "transitive axiom, no sorry"),
    "gt_term_sorry": (True, ["gt_term_sorry"], "term-mode sorry"),
    "gt_uses_term_sorry": (True, ["gt_term_sorry"], "transitive term-mode sorry"),
    "gt_sorry_a": (True, ["gt_sorry_a"], "sorry source A"),
    "gt_sorry_b": (True, ["gt_sorry_b"], "sorry source B"),
    "gt_multi_sorry": (
        True,
        ["gt_sorry_a", "gt_sorry_b"],
        "multiple sorry sources converging",
    ),
    "gt_diamond": (
        True,
        ["gt_shared_sorry"],
        "diamond: shared sorry dep via two paths",
    ),
    "GtNs.ns_clean": (False, [], "namespaced clean theorem"),
    "GtNs.ns_sorry": (True, ["GtNs.ns_sorry"], "namespaced sorry theorem"),
    "GtNs.ns_uses_private": (True, [], "uses private sorry def transitively"),
    "GtOuter.GtInner.nested_sorry": (
        True,
        ["GtOuter.GtInner.nested_sorry"],
        "nested namespace sorry",
    ),
    "gt_in_section": (False, [], "theorem in section, no namespace effect"),
    # 15. Explicit sorry AND sorry'd dependencies
    "gt_sub_sorry_a": (True, ["gt_sub_sorry_a"], "sorry dep A for combined test"),
    "gt_sub_sorry_b": (True, ["gt_sub_sorry_b"], "sorry dep B for combined test"),
    "gt_sorry_with_sorry_deps": (
        True,
        ["gt_sub_sorry_a", "gt_sub_sorry_b"],
        "explicit sorry + sorry'd deps",
    ),
    # 16. Querying a sorry leaf directly
    "gt_leaf_sorry_self": (
        True,
        ["gt_leaf_sorry_self"],
        "direct sorry, queried directly",
    ),
    # 17. Private sorry dep + own sorry
    "GtPrivate.pub_with_priv_dep": (
        True,
        [],
        "public thm with private sorry dep + own sorry",
    ),
    # 18. Inline have sorry
    "gt_inline_have_sorry": (True, ["gt_inline_have_sorry"], "sorry in inline have"),
    # 19. Mixed clean + sorry deps, own sorry
    "gt_mixed_deps": (True, ["gt_sorry_dep"], "mixed clean/sorry deps + own sorry"),
    # 20. Deep chain (depth 3)
    "gt_deep_chain": (True, ["gt_deep_sorry"], "depth-3 sorry chain"),
    # 21. Cross-namespace dependency
    "gt_uses_cross_ns": (True, ["GtCross.cross_sorry"], "cross-namespace sorry dep"),
    # 22. Instance with sorry
    "gt_uses_sorry_instance": (True, ["gt_sorry_instance"], "uses sorry'd instance"),
}


@pytest.mark.asyncio
async def test_verify_matches_ground_truth(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_verify: sorryAx in axioms iff ground truth says sorry."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, _, desc) in GROUND_TRUTH.items():
            result = await client.call_tool(
                "lean_verify",
                {"file_path": str(gt_file), "theorem_name": decl, "scan_source": False},
            )
            data = result_json(result)
            axioms = data["axioms"]
            got_sorry = "sorryAx" in axioms
            assert got_sorry == has_sorry, (
                f"lean_verify {decl} ({desc}): "
                f"expected sorry={has_sorry}, got axioms={axioms}"
            )


@pytest.mark.asyncio
async def test_goal_tracker_matches_ground_truth(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_goal_tracker: sorry_declarations empty iff no sorry in ground truth."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, expected_decls, desc) in GROUND_TRUTH.items():
            result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(gt_file), "decl_name": decl},
            )
            data = result_json(result)
            decl_names = sorry_names(data)

            if has_sorry:
                assert len(decl_names) >= 1, (
                    f"goal_tracker {decl} ({desc}): expected sorry_declarations non-empty, got {data}"
                )
                for ed in expected_decls:
                    assert ed in decl_names, (
                        f"goal_tracker {decl} ({desc}): "
                        f"expected '{ed}' in sorry_declarations={decl_names}"
                    )
            else:
                assert decl_names == [], (
                    f"goal_tracker {decl} ({desc}): expected no sorry, got {data}"
                )

            assert data["total_transitive_deps"] > 0, (
                f"goal_tracker {decl} ({desc}): expected deps > 0, got {data}"
            )


@pytest.mark.asyncio
async def test_goal_tracker_short_name_resolution(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """goal_tracker resolves short names to FQNs via namespace scanning."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        # Short name "ns_sorry" should resolve to GtNs.ns_sorry
        result = await client.call_tool(
            "lean_goal_tracker",
            {"file_path": str(gt_file), "decl_name": "ns_sorry"},
        )
        data = result_json(result)
        assert data["target"] == "ns_sorry"
        assert len(sorry_names(data)) >= 1
        assert data["total_transitive_deps"] > 0


@pytest.mark.asyncio
async def test_goal_tracker_nonexistent_short_name(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """Short name that doesn't exist in the file — error."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_goal_tracker",
            {"file_path": str(gt_file), "decl_name": "totally_nonexistent_xyz"},
            expect_error=True,
        )
        assert result.isError


@pytest.mark.asyncio
async def test_goal_tracker_section_no_namespace_effect(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """Theorem in a section — FQN should not include section name."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "lean_goal_tracker",
            {"file_path": str(gt_file), "decl_name": "gt_in_section"},
        )
        data = result_json(result)
        assert data["target"] == "gt_in_section"
        assert sorry_names(data) == []
        assert data["total_transitive_deps"] > 0


@pytest.mark.asyncio
async def test_verify_and_tracker_agree(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_verify (sorryAx in axioms) should agree with lean_goal_tracker (fully_proven)."""
    gt_file = test_project_path / "GoalTrackerTest.lean"
    async with mcp_client_factory() as client:
        for decl in GROUND_TRUTH:
            verify_result = await client.call_tool(
                "lean_verify",
                {"file_path": str(gt_file), "theorem_name": decl, "scan_source": False},
            )
            tracker_result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(gt_file), "decl_name": decl},
            )
            v_data = result_json(verify_result)
            t_data = result_json(tracker_result)

            v_has_sorry = "sorryAx" in v_data["axioms"]
            t_has_sorry = len(sorry_names(t_data)) > 0

            assert v_has_sorry == t_has_sorry, (
                f"{decl}: verify says sorry={v_has_sorry} "
                f"(axioms={v_data['axioms']}), "
                f"tracker says sorry={t_has_sorry} "
                f"(sorry_decls={t_data['sorry_declarations']})"
            )


# ──────────────────────────────────────────────────────────────────────
# Cross-file tests: GoalTrackerBase.lean + GoalTrackerImport.lean
# ──────────────────────────────────────────────────────────────────────

# Ground truth for GoalTrackerBase.lean (the imported file)
GROUND_TRUTH_BASE = {
    "gtBase_clean": (False, [], "clean def"),
    "gtBase_clean_thm": (False, [], "clean theorem"),
    "gtBase_sorry_val": (True, ["gtBase_sorry_val"], "sorry def"),
    "gtBase_sorry_thm": (True, ["gtBase_sorry_thm"], "sorry theorem"),
    "GtBaseNs.ns_sorry_def": (True, ["GtBaseNs.ns_sorry_def"], "namespaced sorry def"),
    "GtBaseNs.ns_clean_thm": (False, [], "namespaced clean theorem"),
    "GtBasePrivate.pub_uses_priv": (True, [], "pub def using private sorry"),
    "GtBasePrivate.pub_thm_uses_priv": (True, [], "pub thm using private sorry chain"),
    "gtBase_chain_sorry": (True, ["gtBase_chain_sorry"], "chain sorry root"),
    "gtBase_chain_mid": (True, ["gtBase_chain_sorry"], "chain mid"),
    "gtBase_chain_top": (True, ["gtBase_chain_sorry"], "chain top"),
    "gtBase_sorry_instance": (True, ["gtBase_sorry_instance"], "instance with sorry"),
    "gtBase_noncomputable": (False, [], "noncomputable, no sorry"),
}

# Ground truth for GoalTrackerImport.lean (the importing file)
GROUND_TRUTH_IMPORT = {
    # (has_sorry, expected_sorry_decls_subset, description)
    "gtImport_uses_sorry_val": (
        True,
        ["gtBase_sorry_val"],
        "cross-file sorry dep via rfl",
    ),
    "gtImport_uses_clean": (False, [], "cross-file clean dep"),
    "gtImport_chain": (True, ["gtBase_chain_sorry"], "cross-file chain to sorry"),
    "gtImport_uses_ns_sorry": (
        True,
        ["GtBaseNs.ns_sorry_def"],
        "cross-file namespaced sorry",
    ),
    "gtImport_uses_priv_chain": (True, [], "cross-file private sorry chain"),
    "gtImport_uses_sorry_inst": (
        True,
        ["gtBase_sorry_instance"],
        "cross-file sorry instance",
    ),
    "gtImport_mixed": (
        True,
        ["gtBase_sorry_val", "gtImport_local_sorry"],
        "mixed local + cross-file sorry",
    ),
    "gtImport_fully_clean": (False, [], "fully clean in import file"),
    "gtImport_uses_noncomputable": (False, [], "cross-file noncomputable, no sorry"),
    "gtImport_diamond": (True, ["gtBase_sorry_val"], "diamond to cross-file sorry"),
    "gtImport_deep_chain": (True, ["gtBase_chain_sorry"], "deep cross-file chain"),
    "GtImportNs.ns_uses_base_sorry": (
        True,
        ["gtBase_sorry_val"],
        "namespaced cross-file sorry",
    ),
    "GtImportNs.ns_local_sorry": (
        True,
        ["GtImportNs.ns_local_sorry"],
        "namespaced local sorry",
    ),
    "gtImport_where": (
        True,
        ["gtBase_sorry_val"],
        "where clause with cross-file sorry",
    ),
    "gtImport_let_sorry": (
        True,
        ["gtBase_sorry_val"],
        "let binding referencing sorry'd def",
    ),
}


@pytest.mark.asyncio
async def test_cross_file_verify_base(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_verify on GoalTrackerBase.lean declarations."""
    base_file = test_project_path / "GoalTrackerBase.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, _, desc) in GROUND_TRUTH_BASE.items():
            result = await client.call_tool(
                "lean_verify",
                {
                    "file_path": str(base_file),
                    "theorem_name": decl,
                    "scan_source": False,
                },
            )
            data = result_json(result)
            axioms = data["axioms"]
            got_sorry = "sorryAx" in axioms
            assert got_sorry == has_sorry, (
                f"lean_verify BASE {decl} ({desc}): "
                f"expected sorry={has_sorry}, got axioms={axioms}"
            )


@pytest.mark.asyncio
async def test_cross_file_verify_import(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_verify on GoalTrackerImport.lean declarations."""
    import_file = test_project_path / "GoalTrackerImport.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, _, desc) in GROUND_TRUTH_IMPORT.items():
            result = await client.call_tool(
                "lean_verify",
                {
                    "file_path": str(import_file),
                    "theorem_name": decl,
                    "scan_source": False,
                },
            )
            data = result_json(result)
            axioms = data["axioms"]
            got_sorry = "sorryAx" in axioms
            assert got_sorry == has_sorry, (
                f"lean_verify IMPORT {decl} ({desc}): "
                f"expected sorry={has_sorry}, got axioms={axioms}"
            )


@pytest.mark.asyncio
async def test_cross_file_tracker_base(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """goal_tracker on GoalTrackerBase.lean — sorry deps should be within same file."""
    base_file = test_project_path / "GoalTrackerBase.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, expected_decls, desc) in GROUND_TRUTH_BASE.items():
            result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(base_file), "decl_name": decl},
            )
            data = result_json(result)
            decl_names = sorry_names(data)

            if has_sorry:
                assert len(decl_names) >= 1, (
                    f"goal_tracker BASE {decl} ({desc}): "
                    f"expected sorry_declarations non-empty, got {data}"
                )
                for ed in expected_decls:
                    assert ed in decl_names, (
                        f"goal_tracker BASE {decl} ({desc}): "
                        f"expected '{ed}' in sorry_declarations={decl_names}"
                    )
            else:
                assert decl_names == [], (
                    f"goal_tracker BASE {decl} ({desc}): expected no sorry, got {data}"
                )


@pytest.mark.asyncio
async def test_cross_file_tracker_import(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """goal_tracker on GoalTrackerImport.lean — sorry deps should cross file boundaries."""
    import_file = test_project_path / "GoalTrackerImport.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, expected_decls, desc) in GROUND_TRUTH_IMPORT.items():
            result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(import_file), "decl_name": decl},
            )
            data = result_json(result)
            decl_names = sorry_names(data)

            if has_sorry:
                assert len(decl_names) >= 1, (
                    f"goal_tracker IMPORT {decl} ({desc}): "
                    f"expected sorry_declarations non-empty, got {data}"
                )
                for ed in expected_decls:
                    assert ed in decl_names, (
                        f"goal_tracker IMPORT {decl} ({desc}): "
                        f"expected '{ed}' in sorry_declarations={decl_names}"
                    )
            else:
                assert decl_names == [], (
                    f"goal_tracker IMPORT {decl} ({desc}): "
                    f"expected no sorry, got {data}"
                )


@pytest.mark.asyncio
async def test_cross_file_verify_and_tracker_agree(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """lean_verify and goal_tracker must agree on cross-file declarations."""
    import_file = test_project_path / "GoalTrackerImport.lean"
    async with mcp_client_factory() as client:
        for decl in GROUND_TRUTH_IMPORT:
            verify_result = await client.call_tool(
                "lean_verify",
                {
                    "file_path": str(import_file),
                    "theorem_name": decl,
                    "scan_source": False,
                },
            )
            tracker_result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(import_file), "decl_name": decl},
            )
            v_data = result_json(verify_result)
            t_data = result_json(tracker_result)

            v_has_sorry = "sorryAx" in v_data["axioms"]
            t_has_sorry = len(sorry_names(t_data)) > 0

            assert v_has_sorry == t_has_sorry, (
                f"CROSS-FILE {decl}: verify says sorry={v_has_sorry} "
                f"(axioms={v_data['axioms']}), "
                f"tracker says sorry={t_has_sorry} "
                f"(sorry_decls={t_data['sorry_declarations']})"
            )


# ──────────────────────────────────────────────────────────────────────
# Bug reproduction: BFS walks into Mathlib (should only walk project decls)
# ──────────────────────────────────────────────────────────────────────

# Ground truth for GoalTrackerHeavy.lean
GROUND_TRUTH_HEAVY = {
    "gtHeavy_sorry": (True, ["gtHeavy_sorry"], "heavy Mathlib-type sorry def"),
    "gtHeavy_thm": (True, ["gtHeavy_sorry"], "theorem using heavy sorry"),
    "gtHeavy_chain": (True, ["gtHeavy_sorry"], "chain through heavy sorry"),
    "gtHeavy_matrix_sorry": (True, ["gtHeavy_matrix_sorry"], "matrix sorry def"),
    "gtHeavy_matrix_thm": (True, ["gtHeavy_matrix_sorry"], "uses matrix sorry"),
}

# Maximum acceptable transitive deps.  Project declarations only should be
# single-digit to low-hundreds.  If the BFS walks into Mathlib, the count
# explodes to 10,000+ which is the root cause of heartbeat timeouts on
# real projects (the original bug).
MAX_ACCEPTABLE_DEPS = 500


@pytest.mark.asyncio
async def test_heavy_tracker_does_not_walk_mathlib(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    """goal_tracker must NOT walk into Mathlib — total_transitive_deps should be small.

    This is the root cause of the bug where goal_tracker returns empty results
    on real projects: the BFS walks tens of thousands of Mathlib declarations,
    hits the heartbeat limit, and produces no output.
    """
    heavy_file = test_project_path / "GoalTrackerHeavy.lean"
    async with mcp_client_factory() as client:
        for decl, (has_sorry, expected_decls, desc) in GROUND_TRUTH_HEAVY.items():
            result = await client.call_tool(
                "lean_goal_tracker",
                {"file_path": str(heavy_file), "decl_name": decl},
            )
            data = result_json(result)
            decl_names = sorry_names(data)

            # Check sorry correctness
            if has_sorry:
                assert len(decl_names) >= 1, (
                    f"goal_tracker HEAVY {decl} ({desc}): "
                    f"expected sorry_declarations non-empty, got {data}"
                )
                for ed in expected_decls:
                    assert ed in decl_names, (
                        f"goal_tracker HEAVY {decl} ({desc}): "
                        f"expected '{ed}' in sorry_declarations={decl_names}"
                    )

            # THE BUG: BFS walks into Mathlib, causing 10,000+ deps
            assert data["total_transitive_deps"] <= MAX_ACCEPTABLE_DEPS, (
                f"goal_tracker HEAVY {decl} ({desc}): "
                f"total_transitive_deps={data['total_transitive_deps']} exceeds "
                f"max {MAX_ACCEPTABLE_DEPS} — BFS is walking into Mathlib!"
            )
