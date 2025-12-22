#!/usr/bin/env python3
"""
Quick MCP status test - run after restarting Claude Code to verify everything works.

Usage:
    uv run scripts/test_mcp_status.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_PROJECT = PROJECT_ROOT / "tests" / "test_project"


def test_server_import():
    """Test that server imports correctly."""
    print("Testing server import...", end=" ")
    try:
        from lean_lsp_mcp.server import mcp  # noqa: F401
        from lean_lsp_mcp.syntax_utils import get_macro_expansion_at_position  # noqa: F401
        from lean_lsp_mcp.models import MacroExpansion  # noqa: F401

        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_syntax_utils():
    """Test syntax utils module."""
    print("Testing syntax_utils...", end=" ")
    try:
        from lean_lsp_mcp.syntax_utils import _parse_range, _position_in_range

        # Test _parse_range
        range_info = {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10},
        }
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
            ["uv", "run", "pytest", "tests/unit/test_syntax_utils.py", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print("✓ OK")
            return True
        else:
            print("✗ FAILED")
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
            cwd=TEST_PROJECT,
        )
        if "error:" not in result.stderr or "Build completed" in result.stderr:
            print("✓ OK")
            return True
        else:
            print("✗ FAILED")
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
    }

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    print(f"\nStatus written to {status_file}")
    return status


def main():
    print("=" * 50)
    print("lean-lsp-mcp Status Check")
    print("=" * 50)
    print()

    results = {
        "server_import": test_server_import(),
        "syntax_utils": test_syntax_utils(),
        "unit_tests": test_unit_tests(),
        "lean_build": test_lean_project_builds(),
    }

    print()
    print("=" * 50)
    status = write_status_file(results)

    print()
    if status["all_passed"]:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")

    return 0 if status["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
