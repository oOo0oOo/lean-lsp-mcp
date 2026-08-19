"""End-to-end coverage for stale transitive Lean imports."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager, Any

import pytest

from tests.helpers.mcp_client import MCPClient


def _write_project(project: Path, toolchain: str) -> Path:
    (project / "lean-toolchain").write_text(toolchain, encoding="utf-8")
    (project / "lakefile.toml").write_text(
        'name = "stale-import-test"\n'
        'version = "0.1.0"\n'
        'defaultTargets = ["StaleImport"]\n'
        "\n"
        "[[lean_lib]]\n"
        'name = "StaleImport"\n',
        encoding="utf-8",
    )
    sources = project / "StaleImport"
    sources.mkdir()
    dependency = sources / "B.lean"
    dependency.write_text("def value : Nat := 1\n", encoding="utf-8")
    (sources / "C.lean").write_text(
        "import StaleImport.B\n\ndef viaC : Nat := value\n", encoding="utf-8"
    )
    importer = sources / "A.lean"
    importer.write_text(
        "import StaleImport.C\n\nexample : viaC = 1 := rfl\n", encoding="utf-8"
    )
    return importer


def _diagnostic_items(result: Any) -> list[dict[str, Any]]:
    structured = result.structured_content
    assert structured is not None
    if isinstance(structured.get("result"), dict):
        structured = structured["result"]
    return structured["items"]


async def _wait_for_diagnostics(
    client: MCPClient,
    importer: Path,
    predicate: Callable[[list[dict[str, Any]]], bool],
    *,
    timeout: float = 10,
) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await client.call_tool(
            "lean_diagnostic_messages", {"file_path": str(importer)}
        )
        items = _diagnostic_items(result)
        assert not any("Imports are out of date" in item["message"] for item in items)
        if predicate(items):
            return items
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"diagnostics did not converge: {items}")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_transitive_dependency_change_rebuilds_through_mcp(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
    tmp_path: Path,
) -> None:
    """The MCP diagnostic tool rebuilds stale transitive imports once."""
    toolchain = (test_project_path / "lean-toolchain").read_text(encoding="utf-8")
    importer = _write_project(tmp_path, toolchain)
    build = subprocess.run(
        ["lake", "build", "StaleImport.A"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    dependency = tmp_path / "StaleImport" / "B.lean"
    async with mcp_client_factory() as client:
        await _wait_for_diagnostics(client, importer, lambda items: not items)

        dependency.write_text("def value : Nat := 2\n", encoding="utf-8")
        changed = await _wait_for_diagnostics(
            client,
            importer,
            lambda items: any(item["severity"] == "error" for item in items),
        )
        assert any("rfl" in item["message"] for item in changed)

        dependency.write_text("def value : Nat := 1\n", encoding="utf-8")
        await _wait_for_diagnostics(client, importer, lambda items: not items)
