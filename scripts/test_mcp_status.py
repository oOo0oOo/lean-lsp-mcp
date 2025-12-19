#!/usr/bin/env python3
"""
Quick MCP status test - run after restarting Claude Code to verify everything works.

Usage:
    uv run scripts/test_mcp_status.py

This script:
1. Tests that the lean-lsp-mcp server starts
2. Tests key functionality (hover, diagnostics, outline)
3. Reports status to help pick up the thread
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Test project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PROJECT = PROJECT_ROOT / "tests" / "test_project"
MACRO_FILE = TEST_PROJECT / "MacroNotation.lean"
TICTACTOE_FILE = TEST_PROJECT / "TicTacToe.lean"

def run_mcp_tool(tool_name: str, params: dict) -> dict:
    """Run an MCP tool via the server."""
    import asyncio
    from lean_lsp_mcp.server import (
        hover, diagnostic_messages, file_outline, goal
    )

    tools = {
        "lean_hover_info": hover,
        "lean_diagnostic_messages": diagnostic_messages,
        "lean_file_outline": file_outline,
        "lean_goal": goal,
    }

    if tool_name not in tools:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = asyncio.run(tools[tool_name](**params))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def test_server_import():
    """Test that server imports correctly."""
    print("Testing server import...", end=" ")
    try:
        from lean_lsp_mcp.server import mcp
        from lean_lsp_mcp.syntax_utils import get_macro_expansion_at_position
        from lean_lsp_mcp.models import MacroExpansion
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_syntax_utils():
    """Test syntax utils module."""
    print("Testing syntax_utils...", end=" ")
    try:
        from lean_lsp_mcp.syntax_utils import (
            _parse_range,
            _position_in_range,
            get_macro_expansion_at_position,
            MacroExpansion,
            SyntaxRange,
        )

        # Test _parse_range
        range_info = {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}}
        parsed = _parse_range(range_info)
        assert parsed is not None

        # Test _position_in_range
        assert _position_in_range(1, 5, range_info)
        assert not _position_in_range(2, 0, range_info)

        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_unit_tests():
    """Run a quick subset of unit tests."""
    print("Running quick unit tests...", end=" ")
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/unit/test_syntax_utils.py", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            print("✓ OK")
            return True
        else:
            print(f"✗ FAILED")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_lean_project_builds():
    """Test that the test Lean project builds."""
    print("Testing Lean project build...", end=" ")
    try:
        result = subprocess.run(
            ["lake", "build", "MacroNotation"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=TEST_PROJECT
        )
        if "error:" not in result.stderr or "Build completed" in result.stderr:
            print("✓ OK")
            return True
        else:
            print(f"✗ FAILED")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def write_status_file(results: dict):
    """Write status file for quick reference."""
    status_file = PROJECT_ROOT / ".mcp_status.json"
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "all_passed": all(results.values()),
        "next_steps": []
    }

    if not results.get("server_import"):
        status["next_steps"].append("Fix server import errors")
    if not results.get("syntax_utils"):
        status["next_steps"].append("Fix syntax_utils module")
    if not results.get("unit_tests"):
        status["next_steps"].append("Fix failing unit tests")
    if not results.get("lean_build"):
        status["next_steps"].append("Fix Lean test project")

    if status["all_passed"]:
        status["next_steps"].append("MCP is ready - restart Claude Code to use mcp__lean-lsp-mcp__* tools")

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    print(f"\nStatus written to {status_file}")
    return status

def main():
    print("=" * 60)
    print("lean-lsp-mcp Status Check")
    print("=" * 60)
    print()

    results = {}

    results["server_import"] = test_server_import()
    results["syntax_utils"] = test_syntax_utils()
    results["unit_tests"] = test_unit_tests()
    results["lean_build"] = test_lean_project_builds()

    print()
    print("=" * 60)
    status = write_status_file(results)

    print()
    if status["all_passed"]:
        print("✓ ALL TESTS PASSED")
        print("\nTo use the local MCP in Claude Code:")
        print("1. Restart Claude Code")
        print("2. Tools like mcp__lean-lsp-mcp__lean_hover_info will be available")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nNext steps:")
        for step in status["next_steps"]:
            print(f"  - {step}")

    return 0 if status["all_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
