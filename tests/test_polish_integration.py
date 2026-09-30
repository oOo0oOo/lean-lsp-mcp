"""Exercise the revised output contracts against a real Mathlib server."""

from pathlib import Path

import pytest

from tests.helpers.mcp_client import result_json


@pytest.mark.asyncio
async def test_large_goal_and_diagnostic_output(mcp_client_factory, test_project_path):
    path = test_project_path / "LargeGoalOutput.lean"
    names = " ".join(f"long_hypothesis_for_output_budget_{i}" for i in range(300))
    path.write_text(
        "import Mathlib\n"
        "set_option linter.unusedVariables false\n"
        f"theorem large_output_goal ({names} : Nat) : True := by\n"
        "  sorry\n",
        encoding="utf-8",
    )
    try:
        async with mcp_client_factory() as client:
            assert "lean_state_search" not in await client.list_tools()
            args = {"file_path": str(path), "line": 4, "column": 3}
            raw = result_json(await client.call_tool("lean_goal", args))
            structured = result_json(
                await client.call_tool("lean_goal", {**args, "format": "structured"})
            )
            assert raw["status"] == "goals"
            assert len(raw["goals"]) == 1
            assert "characters elided" in raw["goals"][0]
            assert raw["goals"][0].endswith("⊢ True")
            full = structured["goals"][0]
            assert full["goal"] == "True"
            assert len(full["pretty"]) > 6000
            assert len(raw["goals"][0]) < len(full["pretty"])
            diagnostics = result_json(
                await client.call_tool(
                    "lean_diagnostic_messages", {"file_path": str(path)}
                )
            )
            assert any(
                "sorry" in item["message"] and item["category"] == "sorry"
                for item in diagnostics["items"]
            )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_live_axiom_trust(mcp_client_factory, test_project_path: Path):
    path = test_project_path / "AxiomTrustOutput.lean"
    path.write_text(
        "import Mathlib\n"
        "theorem trust_standard : True := trivial\n"
        "theorem trust_incomplete : False := by sorry\n"
        "theorem trust_native : 1 + 1 = 2 := by native_decide\n"
        "axiom custom_truth : False\n"
        "theorem trust_custom : False := custom_truth\n",
        encoding="utf-8",
    )
    try:
        async with mcp_client_factory() as client:
            for trust in ("standard", "incomplete", "native", "custom"):
                data = result_json(
                    await client.call_tool(
                        "lean_verify",
                        {"file_path": str(path), "theorem_name": f"trust_{trust}"},
                    )
                )
                assert data["trust"] == trust, data
                assert bool(data["non_standard_axioms"]) == (trust != "standard")
    finally:
        path.unlink(missing_ok=True)
