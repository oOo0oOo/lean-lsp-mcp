"""Direct correctness test: run the new goal_tracker snippet on all test cases
and compare with #print axioms ground truth.

Run with:
    .venv/bin/python tests/test_snippet_correctness.py
"""

import re
import time
from pathlib import Path

from leanclient import LeanLSPClient, DocumentContentChange

# Import the actual snippet from the codebase
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lean_lsp_mcp.goal_tracker import make_sorry_snippet, parse_sorry_result


TEST_PROJECT = Path(__file__).parent / "test_project"

# All test declarations and expected results: (file, decl, has_sorry, expected_sorry_leaves)
TEST_CASES = [
    # --- GoalTrackerTest.lean (single-file) ---
    ("GoalTrackerTest.lean", "gt_clean", False, []),
    ("GoalTrackerTest.lean", "gt_direct_sorry", True, ["gt_direct_sorry"]),
    ("GoalTrackerTest.lean", "gt_helper", False, []),
    ("GoalTrackerTest.lean", "gt_uses_clean_helper", False, []),
    ("GoalTrackerTest.lean", "gt_sorry_def", True, ["gt_sorry_def"]),
    (
        "GoalTrackerTest.lean",
        "gt_transitive_sorry",
        True,
        ["gt_transitive_sorry", "gt_sorry_def"],
    ),
    ("GoalTrackerTest.lean", "gt_level2_sorry", True, ["gt_level2_sorry"]),
    ("GoalTrackerTest.lean", "gt_level1", True, ["gt_level2_sorry"]),
    ("GoalTrackerTest.lean", "gt_chain_sorry", True, ["gt_level2_sorry"]),
    ("GoalTrackerTest.lean", "gt_decidable", False, []),
    ("GoalTrackerTest.lean", "gt_noncomputable", False, []),
    ("GoalTrackerTest.lean", "gt_uses_noncomputable", False, []),
    ("GoalTrackerTest.lean", "gt_term_sorry", True, ["gt_term_sorry"]),
    ("GoalTrackerTest.lean", "gt_uses_term_sorry", True, ["gt_term_sorry"]),
    ("GoalTrackerTest.lean", "gt_sorry_a", True, ["gt_sorry_a"]),
    ("GoalTrackerTest.lean", "gt_sorry_b", True, ["gt_sorry_b"]),
    ("GoalTrackerTest.lean", "gt_multi_sorry", True, ["gt_sorry_a", "gt_sorry_b"]),
    ("GoalTrackerTest.lean", "gt_diamond", True, ["gt_shared_sorry"]),
    ("GoalTrackerTest.lean", "GtNs.ns_clean", False, []),
    ("GoalTrackerTest.lean", "GtNs.ns_sorry", True, ["GtNs.ns_sorry"]),
    ("GoalTrackerTest.lean", "GtNs.ns_uses_private", True, []),  # private sorry dep
    (
        "GoalTrackerTest.lean",
        "GtOuter.GtInner.nested_sorry",
        True,
        ["GtOuter.GtInner.nested_sorry"],
    ),
    ("GoalTrackerTest.lean", "gt_in_section", False, []),
    ("GoalTrackerTest.lean", "gt_sub_sorry_a", True, ["gt_sub_sorry_a"]),
    ("GoalTrackerTest.lean", "gt_sub_sorry_b", True, ["gt_sub_sorry_b"]),
    (
        "GoalTrackerTest.lean",
        "gt_sorry_with_sorry_deps",
        True,
        ["gt_sub_sorry_a", "gt_sub_sorry_b"],
    ),
    ("GoalTrackerTest.lean", "gt_leaf_sorry_self", True, ["gt_leaf_sorry_self"]),
    (
        "GoalTrackerTest.lean",
        "GtPrivate.pub_with_priv_dep",
        True,
        [],
    ),  # private sorry dep
    ("GoalTrackerTest.lean", "gt_inline_have_sorry", True, ["gt_inline_have_sorry"]),
    ("GoalTrackerTest.lean", "gt_mixed_deps", True, ["gt_sorry_dep"]),
    ("GoalTrackerTest.lean", "gt_deep_chain", True, ["gt_deep_sorry"]),
    ("GoalTrackerTest.lean", "gt_uses_cross_ns", True, ["GtCross.cross_sorry"]),
    ("GoalTrackerTest.lean", "gt_uses_sorry_instance", True, ["gt_sorry_instance"]),
    # --- GoalTrackerBase.lean ---
    ("GoalTrackerBase.lean", "gtBase_clean", False, []),
    ("GoalTrackerBase.lean", "gtBase_clean_thm", False, []),
    ("GoalTrackerBase.lean", "gtBase_sorry_val", True, ["gtBase_sorry_val"]),
    ("GoalTrackerBase.lean", "gtBase_sorry_thm", True, ["gtBase_sorry_thm"]),
    ("GoalTrackerBase.lean", "GtBaseNs.ns_sorry_def", True, ["GtBaseNs.ns_sorry_def"]),
    ("GoalTrackerBase.lean", "GtBaseNs.ns_clean_thm", False, []),
    ("GoalTrackerBase.lean", "GtBasePrivate.pub_uses_priv", True, []),
    ("GoalTrackerBase.lean", "GtBasePrivate.pub_thm_uses_priv", True, []),
    ("GoalTrackerBase.lean", "gtBase_chain_sorry", True, ["gtBase_chain_sorry"]),
    ("GoalTrackerBase.lean", "gtBase_chain_mid", True, ["gtBase_chain_sorry"]),
    ("GoalTrackerBase.lean", "gtBase_chain_top", True, ["gtBase_chain_sorry"]),
    ("GoalTrackerBase.lean", "gtBase_sorry_instance", True, ["gtBase_sorry_instance"]),
    ("GoalTrackerBase.lean", "gtBase_noncomputable", False, []),
    # --- GoalTrackerImport.lean (cross-file) ---
    ("GoalTrackerImport.lean", "gtImport_uses_sorry_val", True, ["gtBase_sorry_val"]),
    ("GoalTrackerImport.lean", "gtImport_uses_clean", False, []),
    ("GoalTrackerImport.lean", "gtImport_chain", True, ["gtBase_chain_sorry"]),
    (
        "GoalTrackerImport.lean",
        "gtImport_uses_ns_sorry",
        True,
        ["GtBaseNs.ns_sorry_def"],
    ),
    ("GoalTrackerImport.lean", "gtImport_uses_priv_chain", True, []),
    (
        "GoalTrackerImport.lean",
        "gtImport_uses_sorry_inst",
        True,
        ["gtBase_sorry_instance"],
    ),
    (
        "GoalTrackerImport.lean",
        "gtImport_mixed",
        True,
        ["gtBase_sorry_val", "gtImport_local_sorry"],
    ),
    ("GoalTrackerImport.lean", "gtImport_fully_clean", False, []),
    ("GoalTrackerImport.lean", "gtImport_uses_noncomputable", False, []),
    ("GoalTrackerImport.lean", "gtImport_diamond", True, ["gtBase_sorry_val"]),
    ("GoalTrackerImport.lean", "gtImport_deep_chain", True, ["gtBase_chain_sorry"]),
    (
        "GoalTrackerImport.lean",
        "GtImportNs.ns_uses_base_sorry",
        True,
        ["gtBase_sorry_val"],
    ),
    (
        "GoalTrackerImport.lean",
        "GtImportNs.ns_local_sorry",
        True,
        ["GtImportNs.ns_local_sorry"],
    ),
    ("GoalTrackerImport.lean", "gtImport_where", True, ["gtBase_sorry_val"]),
    ("GoalTrackerImport.lean", "gtImport_let_sorry", True, ["gtBase_sorry_val"]),
    # --- GoalTrackerMiddle.lean (sorry-free intermediate) ---
    ("GoalTrackerMiddle.lean", "gtMiddle_wrap_sorry", True, ["gtBase_sorry_val"]),
    ("GoalTrackerMiddle.lean", "gtMiddle_wrap_clean", False, []),
    ("GoalTrackerMiddle.lean", "gtMiddle_chain", True, ["gtBase_chain_sorry"]),
    ("GoalTrackerMiddle.lean", "gtMiddle_ns_wrap", True, ["GtBaseNs.ns_sorry_def"]),
    ("GoalTrackerMiddle.lean", "gtMiddle_local_clean", False, []),
    ("GoalTrackerMiddle.lean", "gtMiddle_clean_thm", False, []),
    # --- GoalTrackerTop.lean (sorry through sorry-free middle layer) ---
    ("GoalTrackerTop.lean", "gtTop_uses_middle_sorry", True, ["gtBase_sorry_val"]),
    ("GoalTrackerTop.lean", "gtTop_uses_middle_clean", False, []),
    ("GoalTrackerTop.lean", "gtTop_deep_chain", True, ["gtBase_chain_sorry"]),
    ("GoalTrackerTop.lean", "gtTop_ns_chain", True, ["GtBaseNs.ns_sorry_def"]),
    ("GoalTrackerTop.lean", "gtTop_fully_clean", False, []),
    ("GoalTrackerTop.lean", "gtTop_local_sorry", True, ["gtTop_local_sorry"]),
    (
        "GoalTrackerTop.lean",
        "gtTop_mixed",
        True,
        ["gtBase_sorry_val", "gtTop_local_sorry"],
    ),
    ("GoalTrackerTop.lean", "gtTop_diamond", True, ["gtBase_sorry_val"]),
    # --- GoalTrackerHeavy.lean (Mathlib regression) ---
    ("GoalTrackerHeavy.lean", "gtHeavy_sorry", True, ["gtHeavy_sorry"]),
    ("GoalTrackerHeavy.lean", "gtHeavy_thm", True, ["gtHeavy_sorry"]),
    ("GoalTrackerHeavy.lean", "gtHeavy_chain", True, ["gtHeavy_sorry"]),
    ("GoalTrackerHeavy.lean", "gtHeavy_matrix_sorry", True, ["gtHeavy_matrix_sorry"]),
    ("GoalTrackerHeavy.lean", "gtHeavy_matrix_thm", True, ["gtHeavy_matrix_sorry"]),
]


def run_verify(client: LeanLSPClient, rel_path: str, decl: str) -> list[str]:
    """Run #print axioms and return axiom list."""
    original = client.get_file_content(rel_path)
    original_lines = original.split("\n")
    appended_line = len(original_lines)
    snippet = f"\n#print axioms _root_.{decl}\n"
    snippet_lines = snippet.count("\n")

    change = DocumentContentChange(snippet, [appended_line, 0], [appended_line, 0])
    client.update_file(rel_path, [change])
    raw = client.get_diagnostics(
        rel_path, start_line=appended_line, inactivity_timeout=120.0
    )
    diags = list(raw)

    # Restore
    restore = DocumentContentChange(
        "", [appended_line, 0], [appended_line + snippet_lines, 0]
    )
    client.update_file(rel_path, [restore])

    axioms = []
    for d in diags:
        if d.get("severity") != 3:
            continue
        msg = d.get("message", "").replace("\n", " ")
        if m := re.search(r"depends on axioms:\s*\[(.+?)\]", msg):
            axioms.extend(a.strip() for a in m.group(1).split(","))
    return axioms


def run_tracker(client: LeanLSPClient, rel_path: str, decl: str) -> dict:
    """Run goal_tracker snippet and return parsed result."""
    original = client.get_file_content(rel_path)
    original_lines = original.split("\n")
    appended_line = len(original_lines)
    snippet = make_sorry_snippet(decl)
    snippet_lines = snippet.count("\n")

    change = DocumentContentChange(snippet, [appended_line, 0], [appended_line, 0])
    client.update_file(rel_path, [change])
    raw = client.get_diagnostics(
        rel_path, start_line=appended_line, inactivity_timeout=120.0
    )
    diags = list(raw)

    # Check for errors
    errors = [d.get("message", "") for d in diags if d.get("severity") == 1]

    # Restore
    restore = DocumentContentChange(
        "", [appended_line, 0], [appended_line + snippet_lines, 0]
    )
    client.update_file(rel_path, [restore])

    if errors:
        return {"error": errors, "nodes": {}, "visited": 0}

    nodes, visited = parse_sorry_result(diags)
    sorry_leaves = [n for n, node in nodes.items() if node.explicit_sorry]
    return {"sorry_leaves": sorry_leaves, "nodes": nodes, "visited": visited}


def main():
    print(f"Test project: {TEST_PROJECT}")
    print(f"Total test cases: {len(TEST_CASES)}\n")

    client = LeanLSPClient(str(TEST_PROJECT))
    try:
        # Group by file for efficiency
        files = sorted(set(tc[0] for tc in TEST_CASES))
        print(f"Opening {len(files)} files...")
        for f in files:
            client.open_file(f)
            client.get_diagnostics(f, inactivity_timeout=300.0)
        print("All files ready.\n")

        passed = 0
        failed = 0
        errors = 0

        for file, decl, expect_sorry, expect_leaves in TEST_CASES:
            # Wait for any pending re-elaboration from previous restore
            client.get_diagnostics(file, inactivity_timeout=30.0)
            t0 = time.monotonic()
            result = run_tracker(client, file, decl)
            elapsed = time.monotonic() - t0

            if "error" in result:
                print(
                    f"  ERROR  {decl} ({file}) [{elapsed:.1f}s]: {result['error'][:1]}"
                )
                errors += 1
                continue

            got_sorry = len(result["sorry_leaves"]) > 0
            got_leaves = set(result["sorry_leaves"])
            expect_leaf_set = set(expect_leaves)

            # Check: sorry/no-sorry matches
            sorry_ok = got_sorry == expect_sorry
            # Check: expected leaves are subset of found leaves
            leaves_ok = expect_leaf_set.issubset(got_leaves)

            if sorry_ok and leaves_ok:
                print(f"  PASS   {decl} ({file}) [{elapsed:.1f}s]")
                passed += 1
            else:
                reasons = []
                if not sorry_ok:
                    reasons.append(f"sorry: expected={expect_sorry}, got={got_sorry}")
                if not leaves_ok:
                    missing = expect_leaf_set - got_leaves
                    reasons.append(f"missing leaves: {missing}, got: {got_leaves}")
                print(
                    f"  FAIL   {decl} ({file}) [{elapsed:.1f}s]: {'; '.join(reasons)}"
                )
                failed += 1

        print(f"\n{'=' * 60}")
        print(
            f"Results: {passed} passed, {failed} failed, {errors} errors out of {len(TEST_CASES)}"
        )
        if failed == 0 and errors == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")

    finally:
        client.close()


if __name__ == "__main__":
    main()
