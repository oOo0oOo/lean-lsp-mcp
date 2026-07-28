"""Test file caching optimization."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json, result_text


def _cache_content(name: str, value: int) -> str:
    return f"""import Mathlib

def {name} : Nat := {value}

theorem cacheProof : {name} = {value} := by
  sorry
"""


@pytest.fixture()
def cache_test_file(test_project_path: Path) -> Iterator[Path]:
    path = test_project_path / "CacheTest.lean"
    original = path.read_text(encoding="utf-8")
    path.write_text(
        """import Mathlib

def cachedValue : Nat := 42

theorem cachedTheorem : cachedValue = 42 := by rfl
""",
        encoding="utf-8",
    )
    try:
        yield path
    finally:
        path.write_text(original, encoding="utf-8")


@pytest.mark.asyncio
async def test_file_caching(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    cache_test_file: Path,
) -> None:
    """Test file caching: disk changes detected and tools share state correctly."""

    async with mcp_client_factory() as client:
        # Test 1: Multiple tools share file state correctly
        await client.call_tool(
            "lean_diagnostic_messages", {"file_path": str(cache_test_file)}
        )
        await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        hover = await client.call_tool(
            "lean_hover_info",
            {"file_path": str(cache_test_file), "line": 3, "column": 5},
        )
        assert "cachedValue" in result_text(hover)

        # Test 2: Disk changes are detected and reprocessed correctly
        goal1 = await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        result1 = result_text(goal1)
        # With structured goals, completed proof has empty goals_after list
        assert '"goals_after": []' in result1, (
            f"Expected empty goals_after, got: {result1}"
        )

        # Modify file on disk
        cache_test_file.write_text(
            """import Mathlib

def cachedValue : Nat := 42

theorem cachedTheorem : cachedValue = 42 := by sorry
""",
            encoding="utf-8",
        )

        # Verify change is detected
        goal2 = await client.call_tool(
            "lean_goal", {"file_path": str(cache_test_file), "line": 5}
        )
        result2 = result_text(goal2)

        assert "cachedValue = 42" in result2, (
            f"Should show goal at sorry, got: {result2}"
        )


@pytest.mark.asyncio
async def test_outline_cache_invalidates_during_frequent_external_edits(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    cache_test_file: Path,
) -> None:
    """Outline caching must never hide the latest on-disk file contents."""

    first = _cache_content("cacheOne", 11)
    second = _cache_content("cacheTwo", 22)
    external = _cache_content("cacheSix", 66)
    assert len(first.encode()) == len(second.encode()) == len(external.encode())
    cache_test_file.write_text(first, encoding="utf-8")

    async with mcp_client_factory() as client:
        # Start the LSP before the in-flight test so the edit lands while outline
        # generation is doing its slower scratch-document work.
        await client.call_tool(
            "lean_diagnostic_messages", {"file_path": str(cache_test_file)}
        )
        in_flight = asyncio.create_task(
            client.call_tool("lean_file_outline", {"file_path": str(cache_test_file)})
        )
        await asyncio.sleep(0.2)
        cache_test_file.write_text(second, encoding="utf-8")
        await in_flight

        after_midflight_edit = result_json(
            await client.call_tool(
                "lean_file_outline", {"file_path": str(cache_test_file)}
            )
        )
        names = [item["name"] for item in after_midflight_edit["declarations"]]
        assert "cacheTwo" in names
        assert "cacheOne" not in names

        # An external writer replaces the file with the same byte length and
        # restores its mtime. The content digest must still invalidate the cache.
        previous = cache_test_file.stat()
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys; "
                    "p = pathlib.Path(sys.argv[1]); "
                    "p.write_text(sys.argv[2], encoding='utf-8'); "
                    "os.utime(p, ns=(int(sys.argv[3]), int(sys.argv[4])))"
                ),
                str(cache_test_file),
                external,
                str(previous.st_atime_ns),
                str(previous.st_mtime_ns),
            ],
            check=True,
        )
        current = cache_test_file.stat()
        assert current.st_size == previous.st_size
        assert current.st_mtime_ns == previous.st_mtime_ns

        after_external_edit = result_json(
            await client.call_tool(
                "lean_file_outline", {"file_path": str(cache_test_file)}
            )
        )
        names = [item["name"] for item in after_external_edit["declarations"]]
        assert "cacheSix" in names
        assert "cacheTwo" not in names

        # Rapid edit-then-query cycles exercise LSP document resync as well as
        # outline cache invalidation under the normal agent editing pattern.
        for name, value in [("cacheRed", 33), ("cacheTan", 44), ("cacheSky", 55)]:
            cache_test_file.write_text(_cache_content(name, value), encoding="utf-8")
            goal = result_text(
                await client.call_tool(
                    "lean_goal",
                    {"file_path": str(cache_test_file), "line": 6},
                )
            )
            assert f"{name} = {value}" in goal
